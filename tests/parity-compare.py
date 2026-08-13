#!/usr/bin/env python3
# parity-compare.py: relative RMS and max abs diff between two raw f32 dumps
# Prints one [Parity] line per comparison, --argmax adds the argmax check
# (fails when the argmaxes differ). Exit code 1 when rel RMS exceeds --max-rel.

import argparse
import sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("ref")
    ap.add_argument("got")
    ap.add_argument("--argmax", action="store_true")
    ap.add_argument("--max-rel", type=float, default=5e-2)
    args = ap.parse_args()

    ref = np.fromfile(args.ref, dtype=np.float32)
    got = np.fromfile(args.got, dtype=np.float32)
    if ref.size != got.size:
        print("[Parity] %s FAIL size %d vs %d" % (args.label, ref.size, got.size))
        return 1

    rel = float(np.sqrt(((ref - got) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-20))
    mx = float(np.abs(ref - got).max())
    line = "[Parity] %s: rel rms %.3e max abs %.3e" % (args.label, rel, mx)
    ok = rel <= args.max_rel
    if args.argmax:
        a, b = int(ref.argmax()), int(got.argmax())
        line += " argmax %d vs %d" % (a, b)
        ok = ok and a == b
    print(line + (" OK" if ok else " FAIL"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
