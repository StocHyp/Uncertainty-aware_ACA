"""
Generate samples from a trained triangular transport map.

Loads the .pt saved by train.py, draws z ~ N(0, I), pushes through the
inverse map, un-normalizes, and writes a numpy array.

Includes a sanity check that warns loudly if generated values are wildly
out of scale, which is the signature of an overfit model.

Usage:
    python sample.py --checkpoint ./out/transport_map.pt \
                     --n_samples 1000000 \
                     --out posterior_samples.npy
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from transport_map import MAF
from transforms import to_bounded
from utils import pick_device, save_samples


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MAF(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    norm_mean = ckpt["norm_mean"].cpu().numpy()
    norm_std = ckpt["norm_std"].cpu().numpy()
    # Older checkpoints (pre-bounded support) won't have the mask
    if "bounded_mask" in ckpt:
        bounded_mask = ckpt["bounded_mask"].cpu().numpy().astype(bool)
    else:
        bounded_mask = np.zeros(model.dim, dtype=bool)
    return model, norm_mean, norm_std, bounded_mask


def sample_in_batches(model: MAF, n_samples: int, batch_size: int,
                      device: torch.device) -> np.ndarray:
    parts = []
    done = 0
    while done < n_samples:
        bs = min(batch_size, n_samples - done)
        x = model.sample(bs, device=device)
        parts.append(x.cpu().numpy())
        done += bs
        if n_samples >= 50_000:
            print(f"  sampled {done}/{n_samples}", flush=True)
    return np.concatenate(parts, axis=0)


def sanity_check(samples: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Warn if generated samples are clearly out of distribution.

    The signature of an overfit MAF is samples with magnitudes orders of
    magnitude larger than the training data. We use the per-dim std as
    the natural scale and flag if any sample is more than 50 standard
    deviations from the per-dim mean.
    """
    z_score = np.abs((samples - mean) / np.maximum(std, 1e-8))
    max_z = z_score.max()
    frac_extreme = (z_score > 10).mean()

    if not np.all(np.isfinite(samples)):
        print("[FAIL] generated samples contain NaN or Inf -- model is broken")
        return False
    if max_z > 50:
        print(f"[FAIL] max z-score across all samples and dims is {max_z:.1e}")
        print(f"       this almost certainly means the model is overfit;")
        print(f"       retrain with stronger regularization or more thinning")
        return False
    if frac_extreme > 0.01:
        print(f"[warn] {frac_extreme:.2%} of values are >10 sigma from "
              f"per-dim mean (max z={max_z:.1f}); check diagnose.py output")
        return True
    print(f"[ok] sanity check passed (max z={max_z:.1f}, "
          f"extreme frac={frac_extreme:.4%})")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_samples", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=10_000)
    p.add_argument("--out", default="generated_samples.npy")
    p.add_argument("--device", default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"[info] device: {device}")

    model, mean, std, bounded_mask = load_model(Path(args.checkpoint), device)
    print(f"[info] loaded MAF: {model.config}")
    n_bounded = int(bounded_mask.sum())
    if n_bounded > 0:
        print(f"[info] will apply sigmoid + clip to {n_bounded}/{model.dim} "
              f"bounded dims at output")

    print(f"[info] sampling {args.n_samples} points...")
    z_samples = sample_in_batches(model, args.n_samples, args.batch_size, device)

    # Un-standardize back to logit-space, then apply sigmoid + clip on
    # bounded dims to recover the [0, 1] range.
    samples = z_samples * std + mean
    if n_bounded > 0:
        samples = to_bounded(samples, bounded_mask, clip=True)
        # In bounded mode, the only meaningful sanity check is that the
        # values lie in [0, 1] -- which clipping already guarantees -- and
        # are finite. The unbounded dims still get the z-score check.
        if (~bounded_mask).any():
            sanity_check(samples[:, ~bounded_mask],
                         mean[~bounded_mask], std[~bounded_mask])
        elif not np.all(np.isfinite(samples)):
            print("[FAIL] generated samples contain NaN or Inf")
        else:
            print(f"[ok] sanity check passed (all dims bounded; "
                  f"clipped to [0, 1])")
    else:
        sanity_check(samples, mean, std)
    save_samples(samples, args.out)


if __name__ == "__main__":
    main()
