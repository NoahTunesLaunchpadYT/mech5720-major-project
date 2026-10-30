"""Reconstruct a 4x4 depth image from a sliding coded-mask radar scan.

Run from the project root:
    python demux.py

The default input is captures/02_right_sweep and its matching ambient scan is
captures/ambient.  Results are saved beside the input scan as demux.npz and
demux_heatmap.png.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
MASK_FILE = ROOT / "mask_patterns" / "mask_pattern.csv"


def load_scan(folder):
    """Return the per-mask mean intensity and distance axis saved by radar_capture.py."""
    with np.load(folder / "scene_map.npz") as data:
        return data["scene"].astype(float), data["distances_m"].astype(float)


def decode(scene, ambient, mask):
    """Ambient-subtract then least-squares-decode every distance bin into 4x4 pixels."""
    if scene.shape != ambient.shape or scene.shape[0] != 24:
        raise ValueError("scene and ambient must both contain 24 matching mask profiles")
    if mask.shape != (4, 27):
        raise ValueError("mask_pattern.csv must be 4 x 27")
    if not np.isfinite(scene).all() or not np.isfinite(ambient).all():
        raise ValueError("all 24 scene and ambient masks must be captured")

    # Row m describes the 4x4 aperture shown for capture mask m.
    aperture = np.array([mask[:, m:m + 4].ravel() for m in range(24)], dtype=float)
    pixels, *_ = np.linalg.lstsq(aperture, scene - ambient, rcond=None)
    return pixels.reshape(4, 4, -1)


def depth_image(reconstruction, distances_m):
    """Return per-pixel peak depth and strength, suppressing weak noise-only cells."""
    positive = np.maximum(reconstruction, 0)
    peak_index = positive.argmax(axis=2)
    strength = positive.max(axis=2)
    noise = np.median(np.abs(reconstruction - np.median(reconstruction)))
    threshold = max(5 * noise, positive.max() * 1e-6, np.finfo(float).eps)
    depth_m = distances_m[peak_index].astype(float)
    depth_m[strength < threshold] = np.nan
    return depth_m, strength, threshold


def save_heatmap(depth_m, strength, output):
    if not np.isfinite(depth_m).any():
        raise ValueError("no reconstructed return exceeds the noise threshold")
    cmap = plt.colormaps["turbo"].copy()
    cmap.set_bad("#202020")
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    image = ax.imshow(depth_m, cmap=cmap, vmin=np.nanmin(depth_m), vmax=np.nanmax(depth_m))
    ax.set(title="Coded-mask radar depth reconstruction", xlabel="Mask column", ylabel="Mask row",
           xticks=range(4), yticks=range(4))
    for row, col in np.ndindex(depth_m.shape):
        if np.isfinite(depth_m[row, col]):
            ax.text(col, row, f"{strength[row, col]:.2g}", ha="center", va="center",
                    color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="Predicted depth (m)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def self_test():
    mask = np.loadtxt(MASK_FILE, delimiter=",", dtype=int)
    aperture = np.array([mask[:, m:m + 4].ravel() for m in range(24)])
    source = np.zeros((16, 3))
    source[6, 1] = 10
    reconstruction = decode(aperture @ source, np.zeros((24, 3)), mask)
    depth_m, _, _ = depth_image(reconstruction, np.array([0.5, 1.0, 1.5]))
    assert np.isclose(depth_m[1, 2], 1.0) and np.isnan(depth_m).sum() == 15


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", type=Path, default=ROOT / "captures" / "02_right_sweep")
    parser.add_argument("--ambient", type=Path, default=ROOT / "captures" / "ambient")
    parser.add_argument("--show", action="store_true", help="also open the saved heatmap")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("demux self-test passed")
        return

    scene, distances_m = load_scan(args.scene)
    ambient, ambient_distances_m = load_scan(args.ambient)
    if not np.array_equal(distances_m, ambient_distances_m):
        raise ValueError("scene and ambient distance axes differ")
    reconstruction = decode(scene, ambient, np.loadtxt(MASK_FILE, delimiter=",", dtype=int))
    depth_m, strength, threshold = depth_image(reconstruction, distances_m)
    np.savez(args.scene / "demux.npz", reconstruction=reconstruction, distances_m=distances_m,
             depth_m=depth_m, strength=strength, threshold=threshold)
    output = args.scene / "demux_heatmap.png"
    save_heatmap(depth_m, strength, output)
    print(f"saved {output} (signal threshold {threshold:.3g})")
    if args.show:
        plt.imshow(plt.imread(output))
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
