#!/usr/bin/env python3
"""
visualize_h36m.py

Script to visualize a few images from the Human3.6M (H36M) dataset along with their
2D keypoint annotations overlaid on the image (left panel) and the corresponding
3D keypoint skeleton rendered in a 3D subplot (right panel). Dot radius is set to 1.
The 3D points are rotated –90° about the X-axis, then +90° about the Z-axis.

Usage:
    python visualize_h36m.py

Adjust `DATA_PATH` so it points to your H36M root folder, which must contain:
    - annot/
        • train_images.txt
        • train.h5
        • valid_images.txt
        • valid.h5
    - images/  (all H36M images, possibly in subdirectories)
"""

import os
import random
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Number of samples to visualize in total
NUM_SAMPLES = 5

# ----------------------------------------------------------------------
# TODO: Change this to your local H36M root directory
DATA_PATH = "/home/oliver/Documents/data/h36m"
# ----------------------------------------------------------------------

ANNOT_DIR = os.path.join(DATA_PATH, "annot")
IMAGES_DIR = os.path.join(DATA_PATH, "images")  # H36M images base folder

# Which split to visualize?
SPLIT = "valid"  # or "train"

# Skeleton connectivity (using corrected joint indices):
H36M_SKELETON = [
    # Right leg
    (0, 1), (1, 2), (2, 3),
    # Left leg
    (0, 4), (4, 5), (5, 6),
    # Spine to head chain
    (0, 7), (7, 8), (8, 9), (9, 10),
    # Left arm chain
    (11, 12), (12, 13),
    # Right arm chain
    (14, 15), (15, 16),
    # Shoulders to thorax
    (11, 8), (8, 14)
]


def build_image_lookup(root_dir):
    """
    Walk through root_dir (and subfolders) and build a dict mapping each image basename
    to its absolute path. This handles cases where H36M images live in subdirectories.
    """
    lookup = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                lookup[fname] = os.path.join(dirpath, fname)
    return lookup


def load_image_list_and_annotations(split="train"):
    """
    Read <split>_images.txt to get the list of image-relative paths, then open the
    corresponding HDF5 (.h5) file to extract raw 2D and 3D joint annotations.

    Returns:
        image_names : [str] list of image paths (relative to DATA_PATH)
        joints_2d   : np.ndarray (N,17,2) of raw 2D keypoints in pixel coords
        joints_3d   : np.ndarray (N,17,3) of raw 3D keypoints (in meters)
    """
    if split == "train":
        img_list_file = os.path.join(ANNOT_DIR, "train_images.txt")
        h5_file       = os.path.join(ANNOT_DIR, "train.h5")
    else:
        img_list_file = os.path.join(ANNOT_DIR, "valid_images.txt")
        h5_file       = os.path.join(ANNOT_DIR, "valid.h5")

    with open(img_list_file, "r") as f:
        image_names = [line.strip() for line in f.readlines()]

    with h5py.File(h5_file, "r") as f:
        raw_2d = f["part"][:]      # possibly shape (N,2,17)
        raw_3d = f["S"][:]         # possibly shape (N,3,17)

    # If raw_2d is (N,2,17), transpose to (N,17,2)
    if raw_2d.ndim == 3 and raw_2d.shape[1] == 2 and raw_2d.shape[2] == 17:
        joints_2d = raw_2d.transpose(0, 2, 1).copy()
    else:
        joints_2d = raw_2d.copy()

    # If raw_3d is (N,3,17), transpose to (N,17,3)
    if raw_3d.ndim == 3 and raw_3d.shape[1] == 3 and raw_3d.shape[2] == 17:
        joints_3d = raw_3d.transpose(0, 2, 1).copy()
    else:
        joints_3d = raw_3d.copy()

    assert (
        joints_2d.shape[0] == joints_3d.shape[0] == len(image_names)
    ), (
        f"Mismatch: 2D has {joints_2d.shape[0]}, 3D has {joints_3d.shape[0]}, "
        f"but txt lists {len(image_names)} images."
    )

    return image_names, joints_2d, joints_3d


def plot_2d_on_axis(ax, image, keypoints, skeleton, radius=1, color=(0, 1, 0), linewidth=2):
    """
    Draw 2D keypoints & skeleton on a given Matplotlib Axes (ax).
      - ax: a 2D axes (imshow)
      - image: HxWx3 RGB numpy array
      - keypoints: (17,2) array of pixel coords
      - skeleton: list of (parent, child) pairs
      - radius: size of each dot (set to 1)
      - color: skeleton color
      - linewidth: skeleton line thickness
    """
    ax.imshow(image)
    ax.axis("off")

    # Plot joints as red circles
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]
    ax.scatter(xs, ys, c="r", s=(radius * 5) ** 2, edgecolors="white", linewidths=0.5)

    # Plot skeleton bones
    for (p, c) in skeleton:
        ax.plot(
            [keypoints[p, 0], keypoints[c, 0]],
            [keypoints[p, 1], keypoints[c, 1]],
            c=color,
            linewidth=linewidth,
        )


def rotate_x_minus90(joints_3d):
    """
    Rotate the 3D points by -90 degrees about the X-axis.
    Rotation: (x, y, z) → (x, z, -y)
    Input:
        joints_3d: (17,3) numpy array
    Returns:
        rotated: (17,3) numpy array
    """
    x = joints_3d[:, 0]
    y = joints_3d[:, 1]
    z = joints_3d[:, 2]
    x_new = x
    y_new = z
    z_new = -y
    return np.stack([x_new, y_new, z_new], axis=1)


def rotate_z_plus90(joints_3d):
    """
    Rotate the 3D points by +90 degrees about the Z-axis.
    Rotation: (x, y, z) → (-y, x, z)
    Input:
        joints_3d: (17,3) numpy array (already rotated about X if chaining)
    Returns:
        rotated: (17,3) numpy array
    """
    x = joints_3d[:, 0]
    y = joints_3d[:, 1]
    z = joints_3d[:, 2]
    x_new = -y
    y_new = x
    z_new = z
    return np.stack([x_new, y_new, z_new], axis=1)


def rotate_combined(joints_3d):
    """
    Apply first rotation by -90° about X, then +90° about Z.
    Input:
        joints_3d: (17,3) numpy array
    Returns:
        rotated: (17,3) numpy array
    """
    rotated_x = rotate_x_minus90(joints_3d)
    return rotated_x


def plot_3d_on_axis(ax3d, joints_3d, skeleton, title=None):
    """
    Draw 3D skeleton on an existing 3D axis (ax3d), after applying rotation.
    """
    # Convert to millimeters if in meters, then center at pelvis (joint 0)
    if np.max(np.abs(joints_3d)) < 10:
        joints_3d = joints_3d * 1000.0
    joints_centered = joints_3d - joints_3d[0]

    # Apply both rotations: first X, then Z
    rotated = rotate_combined(joints_centered)

    # Scatter joints
    ax3d.scatter(rotated[:, 0], rotated[:, 1], rotated[:, 2], c="r", s=30)

    # Plot bones
    for (p, c) in skeleton:
        ax3d.plot(
            [rotated[p, 0], rotated[c, 0]],
            [rotated[p, 1], rotated[c, 1]],
            [rotated[p, 2], rotated[c, 2]],
            c="b",
            linewidth=2,
        )

    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    if title:
        ax3d.set_title(title)

    # Equal aspect ratio
    xyz_min = np.min(rotated, axis=0)
    xyz_max = np.max(rotated, axis=0)
    max_range = np.max(xyz_max - xyz_min) / 2.0
    mid_x = (xyz_max[0] + xyz_min[0]) / 2.0
    mid_y = (xyz_max[1] + xyz_min[1]) / 2.0
    mid_z = (xyz_max[2] + xyz_min[2]) / 2.0

    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)


def main():
    # Build a lookup: basename → full path. Handles subfolders under IMAGES_DIR.
    print("Building image lookup dictionary…")
    image_lookup = build_image_lookup(IMAGES_DIR)

    # Load H36M image list + annotations for the chosen split
    print(f"Loading H36M {SPLIT} split annotations…")
    image_names, joints_2d_all, joints_3d_all = load_image_list_and_annotations(SPLIT)

    # Randomly select which indices to visualize
    num_samples = min(NUM_SAMPLES, len(image_names))
    chosen_indices = random.sample(range(len(image_names)), num_samples)

    for idx in chosen_indices:
        relpath = image_names[idx]
        basename = os.path.basename(relpath)
        fullpath = image_lookup.get(basename)

        if fullpath is None or not os.path.isfile(fullpath):
            print(f"WARNING: cannot find image for '{relpath}'. Skipping index {idx}.")
            continue

        # Load BGR image with OpenCV, convert to RGB
        img_bgr = cv2.imread(fullpath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Get the raw 2D keypoints (pixels) for this index
        kpts2d = joints_2d_all[idx].astype(np.float32)  # shape (17,2)

        # Get the raw 3D keypoints (meters)
        kpts3d = joints_3d_all[idx].astype(np.float32)  # shape (17,3)

        # Create one figure with two subplots: (1, 2)
        fig = plt.figure(figsize=(16, 8))

        # --- Left subplot: 2D overlay (dot radius = 1) ---
        ax2d = fig.add_subplot(1, 2, 1)
        plot_2d_on_axis(
            ax2d,
            img_rgb,
            kpts2d,
            H36M_SKELETON,
            radius=1,            # dot radius = 1
            color=(0, 1, 0),     # green skeleton lines
            linewidth=2,
        )
        ax2d.set_title(f"{SPLIT.capitalize()} idx={idx} — 2D Keypoints")

        # --- Right subplot: 3D skeleton (rotated –90° about X, then +90° about Z) ---
        ax3d = fig.add_subplot(1, 2, 2, projection="3d")
        plot_3d_on_axis(
            ax3d,
            kpts3d,
            H36M_SKELETON,
            title=f"{SPLIT.capitalize()} idx={idx} — 3D Keypoints (rotated)",
        )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
