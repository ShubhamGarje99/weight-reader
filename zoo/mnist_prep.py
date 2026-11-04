"""Parse raw MNIST IDX files and project to 16 PCA components (standardised).

Usage: python -I zoo/mnist_prep.py data.noindex/mnist_raw data.noindex/mnist_pca16.npz
"""
import sys
import numpy as np


def read_idx(path):
    with open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        ndim = magic & 0xFF
        shape = [int.from_bytes(f.read(4), "big") for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(shape)


def main(raw_dir, out):
    xtr = read_idx(f"{raw_dir}/train-images-idx3-ubyte").reshape(-1, 784).astype(np.float32) / 255.0
    ytr = read_idx(f"{raw_dir}/train-labels-idx1-ubyte").astype(np.int64)
    xte = read_idx(f"{raw_dir}/t10k-images-idx3-ubyte").reshape(-1, 784).astype(np.float32) / 255.0
    yte = read_idx(f"{raw_dir}/t10k-labels-idx1-ubyte").astype(np.int64)
    assert xtr.shape == (60000, 784) and xte.shape == (10000, 784)
    mean = xtr.mean(0)
    rng = np.random.default_rng(0)
    sub = xtr[rng.choice(60000, 20000, replace=False)] - mean
    # PCA via SVD on a 20k subset (keeps memory small)
    _, _, vt = np.linalg.svd(sub, full_matrices=False)
    comps = vt[:16]
    ftr = (xtr - mean) @ comps.T
    fte = (xte - mean) @ comps.T
    sd = ftr.std(0) + 1e-6
    ftr = (ftr / sd).astype(np.float32)
    fte = (fte / sd).astype(np.float32)
    np.savez_compressed(out, x_train=ftr, y_train=ytr, x_test=fte, y_test=yte, components=comps, mean=mean, sd=sd)
    print("saved", out, ftr.shape, fte.shape, "feature std", ftr.std(0).round(2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
