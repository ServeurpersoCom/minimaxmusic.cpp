#!/usr/bin/env python3
# Training loop for the RVQ encoder, hiddens route: AdamW, bf16
# autocast, MSE plus cosine loss against the dumped LM hidden states.
# Several corpora can be given at once, so a run scales by adding
# datasets rather than by regenerating one. The validation split holds
# out whole tracks, one prompt in ten for corpora named p<prompt>-r<...>
# and the last song of each batch otherwise, so validation always sees
# material the model never trained on. Checkpoints land in
# training/checkpoints/<run>/ as last.pt, best.pt (lowest validation
# loss) and ema.pt, the mean of the weights over the final epochs, which
# is the one to evaluate: it costs nothing and is steadier than any
# single epoch.
#
# Usage: ./train.py <dataset> [<dataset> ...] [--run NAME] [--epochs N]
#                   [--batch N] [--lr X] [--seed N]

import argparse
import math
import os
import re
import time

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from dataset import VAE_EXT, HID_EXT, HiddenDataset
from model import HiddenEncoder

DATASETS_DIR      = os.path.join(os.path.dirname(__file__), "datasets")
CHECKPOINTS_DIR   = os.path.join(os.path.dirname(__file__), "checkpoints")
VAL_PROMPT_STRIDE = 10
WARMUP_STEPS      = 500
LOG_EVERY_STEPS   = 50
EMA_EPOCHS        = 20


def is_held_out(base: str) -> bool:
    # Corpus bases are p<prompt>-r<round><song><variation>; anything else
    # holds out the last song of each generation batch
    match = re.match(r"^p(\d\d)-r\d\d", base)
    if match:
        return int(match.group(1)) % VAL_PROMPT_STRIDE == VAL_PROMPT_STRIDE - 1
    return base.endswith("50")


def split_bases(corpus_dir: str) -> tuple[list[str], list[str]]:
    bases = sorted(f[: -len(VAE_EXT)] for f in os.listdir(corpus_dir) if f.endswith(VAE_EXT))
    bases = [b for b in bases if os.path.exists(os.path.join(corpus_dir, b + HID_EXT))]
    return ([b for b in bases if not is_held_out(b)],
            [b for b in bases if is_held_out(b)])


def loss_terms(pred, target):
    # MSE matches the magnitude, the cosine term matches the direction
    # the published heads read
    mse = F.mse_loss(pred, target)
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    return mse + cos, mse, cos


def evaluate(model, loader, device):
    model.eval()
    loss_sum, cos_sum, n = 0.0, 0.0, 0
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for latents, pool, target in loader:
            latents, pool, target = latents.to(device), pool.to(device), target.to(device)
            loss, mse, cos = loss_terms(model(latents, pool).float(), target.float())
            loss_sum += loss.item()
            cos_sum  += 1.0 - cos.item()
            n += 1
    model.train()
    return loss_sum / n, cos_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--run", default="v3")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device   = "cuda"
    ckpt_dir = os.path.join(CHECKPOINTS_DIR, args.run)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_sets, val_sets = [], []
    for name in args.datasets:
        corpus_dir = os.path.join(DATASETS_DIR, name)
        train_bases, val_bases = split_bases(corpus_dir)
        train_sets.append(HiddenDataset(corpus_dir, train_bases, random_crop=True))
        val_sets.append(HiddenDataset(corpus_dir, val_bases))
        print(f"[Data] {name}: {len(train_bases)} train tracks ({len(train_sets[-1])} windows), "
              f"{len(val_bases)} val tracks ({len(val_sets[-1])} windows)", flush=True)

    train_set, val_set = ConcatDataset(train_sets), ConcatDataset(val_sets)
    print(f"[Data] total {len(train_set)} train windows, {len(val_set)} val windows", flush=True)

    # The pod caps /dev/shm at 63 MB, below one prefetched batch: the
    # loaders run in the main process
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch, num_workers=0)

    model = HiddenEncoder().to(device)
    print(f"[Model] {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params", flush=True)

    opt         = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / WARMUP_STEPS, 0.5 * (1.0 + math.cos(math.pi * s / total_steps))))

    ema, ema_epochs = None, 0
    best_val = float("inf")
    step     = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        for latents, pool, target in train_loader:
            latents = latents.to(device, non_blocking=True)
            pool    = pool.to(device, non_blocking=True)
            target  = target.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16):
                loss, mse, cos = loss_terms(model(latents, pool).float(), target.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % LOG_EVERY_STEPS == 0:
                print(f"[Train] Epoch {epoch} step {step}/{total_steps}: loss {loss.item():.4f} "
                      f"(mse {mse.item():.4f}, cossim {1.0 - cos.item():.4f}), "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)

        val_loss, val_cos = evaluate(model, val_loader, device)
        print(f"[Val] Epoch {epoch}: loss {val_loss:.4f}, h cossim {val_cos:.4f}, "
              f"{time.time() - t0:.1f} s/epoch", flush=True)

        state = {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}
        torch.save(state, os.path.join(ckpt_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, os.path.join(ckpt_dir, "best.pt"))
            print(f"[Ckpt] best.pt updated (val loss {val_loss:.4f})", flush=True)

        if epoch >= args.epochs - EMA_EPOCHS:
            weights = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
            ema_epochs += 1
            if ema is None:
                ema = weights
            else:
                for k in ema:
                    ema[k] += (weights[k] - ema[k]) / ema_epochs

    torch.save({"model": ema, "epoch": args.epochs - 1, "val_loss": best_val,
                "averaged_epochs": ema_epochs}, os.path.join(ckpt_dir, "ema.pt"))
    print(f"[Ckpt] ema.pt written, mean of the last {ema_epochs} epochs", flush=True)


if __name__ == "__main__":
    main()
