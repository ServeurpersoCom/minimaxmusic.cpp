#!/usr/bin/env python3
# Training loop for the RVQ encoder: AdamW, bf16 autocast, cross-entropy
# averaged over the 8 heads. The validation split holds out whole
# prompts (index % 10 == 9) so validation sees captions the model never
# trained on. Checkpoints land in training/checkpoints/<run>/ as
# last.pt and best.pt (lowest validation loss).
#
# Usage: ./train.py <dataset> [--run NAME] [--epochs N] [--batch N] [--lr X]

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from dataset import VAE_EXT, CorpusDataset, RandomCropDataset
from model import ACOUSTIC_HEADS, RVQEncoder

DATASETS_DIR      = os.path.join(os.path.dirname(__file__), "datasets")
CHECKPOINTS_DIR   = os.path.join(os.path.dirname(__file__), "checkpoints")
VAL_PROMPT_STRIDE = 10
WARMUP_STEPS      = 500
LOG_EVERY_STEPS   = 50
SEED              = 42


def prompt_index(base: str) -> int:
    # Corpus bases are p<prompt>-r<round><song><variation>
    return int(base[1:3])


def split_bases(corpus_dir: str) -> tuple[list[str], list[str]]:
    bases = sorted(f[: -len(VAE_EXT)] for f in os.listdir(corpus_dir) if f.endswith(VAE_EXT))
    train = [b for b in bases if prompt_index(b) % VAL_PROMPT_STRIDE != VAL_PROMPT_STRIDE - 1]
    val   = [b for b in bases if prompt_index(b) % VAL_PROMPT_STRIDE == VAL_PROMPT_STRIDE - 1]
    return train, val


def head_losses(model_out, target):
    # target [B, 128, 8], column 0 semantic then 7 acoustic
    sem_logits, ac_logits = model_out
    losses = [F.cross_entropy(sem_logits.flatten(0, 1), target[:, :, 0].flatten())]
    for i in range(ACOUSTIC_HEADS):
        losses.append(F.cross_entropy(ac_logits[i].flatten(0, 1), target[:, :, i + 1].flatten()))
    return losses


def accuracy(model_out, target):
    sem_logits, ac_logits = model_out
    sem = (sem_logits.argmax(-1) == target[:, :, 0]).float().mean().item()
    ac  = sum((ac_logits[i].argmax(-1) == target[:, :, i + 1]).float().mean().item()
              for i in range(ACOUSTIC_HEADS)) / ACOUSTIC_HEADS
    return sem, ac


def evaluate(model, loader, device):
    model.eval()
    loss_sum, sem_sum, ac_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for latents, pool, target in loader:
            latents, pool, target = latents.to(device), pool.to(device), target.to(device)
            out    = model(latents, pool)
            losses = head_losses(out, target)
            sem, ac = accuracy(out, target)
            loss_sum += sum(l.item() for l in losses) / len(losses)
            sem_sum  += sem
            ac_sum   += ac
            n += 1
    model.train()
    return loss_sum / n, sem_sum / n, ac_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--run", default="v1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device     = "cuda"
    corpus_dir = os.path.join(DATASETS_DIR, args.dataset)
    ckpt_dir   = os.path.join(CHECKPOINTS_DIR, args.run)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_bases, val_bases = split_bases(corpus_dir)
    train_set = RandomCropDataset(corpus_dir, train_bases)
    val_set   = CorpusDataset(corpus_dir, val_bases)
    print(f"[Data] {len(train_bases)} train tracks ({len(train_set)} windows), "
          f"{len(val_bases)} val tracks ({len(val_set)} windows)")

    # The pod caps /dev/shm at 63 MB, below one prefetched batch: the
    # loaders run in the main process, the per-sample work is one small
    # np.fromfile plus a pooling matrix build
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch, num_workers=0)

    model = RVQEncoder().to(device)
    print(f"[Model] {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params")

    opt         = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / WARMUP_STEPS, 0.5 * (1.0 + math.cos(math.pi * s / total_steps))))

    best_val = float("inf")
    step     = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        for latents, pool, target in train_loader:
            latents = latents.to(device, non_blocking=True)
            pool    = pool.to(device, non_blocking=True)
            target  = target.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16):
                losses = head_losses(model(latents, pool), target)
                loss   = sum(losses) / len(losses)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % LOG_EVERY_STEPS == 0:
                print(f"[Train] Epoch {epoch} step {step}/{total_steps}: loss {loss.item():.4f} "
                      f"(sem {losses[0].item():.4f}, ac {sum(l.item() for l in losses[1:]) / ACOUSTIC_HEADS:.4f}), "
                      f"lr {sched.get_last_lr()[0]:.2e}")

        val_loss, val_sem, val_ac = evaluate(model, val_loader, device)
        print(f"[Val] Epoch {epoch}: loss {val_loss:.4f}, sem top1 {val_sem:.3f}, "
              f"ac top1 {val_ac:.3f}, {time.time() - t0:.1f} s/epoch")

        state = {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}
        torch.save(state, os.path.join(ckpt_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, os.path.join(ckpt_dir, "best.pt"))
            print(f"[Ckpt] best.pt updated (val loss {val_loss:.4f})")


if __name__ == "__main__":
    main()
