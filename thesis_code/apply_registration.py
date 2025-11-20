#Apply Registration for co-registration

"""
"""

import os
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat


def apply_manual_rotation(mask: np.ndarray, fixed_shape: tuple[int, int],theta_deg: float, dx: float, dy: float) -> np.ndarray:
  
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)

    fixed_h, fixed_w = fixed_shape

    h, w = mask_u8.shape[:2]
    center = (w // 2, h // 2)

    rot_mat = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    rotated = cv2.warpAffine(mask_u8, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


    trans_mat = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(rotated,trans_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0,)

    resized = cv2.resize(translated, (fixed_w, fixed_h), interpolation=cv2.INTER_NEAREST)

    return resized


def apply_affine(mask: np.ndarray,
                 T: np.ndarray,
                 fixed_shape: tuple[int, int]) -> np.ndarray:

    fixed_h, fixed_w = fixed_shape

    # OpenCV bruker 2x3 matrise
    M = T[:2, :]

    warped = cv2.warpAffine(
        mask,
        M,
        (fixed_w, fixed_h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return warped


def main():

    dir0 = Path("/Volumes/Expansion/Prostata_ST")
    dir_h = dir0 / "Histology"
    dir_f = dir_h / "Co-reg/Files"
    dir_i = dir_f / "Microscopy images"
    dir_r = dir_f / "Rotation matrix"

    name_list = ["Func116"]  

    string = "HE"

    register = ["neoplastic", "connective"]

    dir_digpat = dir_h / f"Digital pathology/Analysis/Files/{string}"
    dir_save = dir_f / f"Feature maps/{string}"
    dir_save.mkdir(parents=True, exist_ok=True)

    for name in name_list:
        print(f"\n=== Processing {name} ({string}) ===")

        fixed_path = dir_i / f"{name} HE Visium.tif"
        img_fixed = cv2.imread(str(fixed_path), cv2.IMREAD_COLOR)
        fixed_h, fixed_w = img_fixed.shape[:2]

        rot_mat_path = dir_r / f"{name} {string}.mat"

        mat = loadmat(rot_mat_path)
        scale_factor = float(mat["scaleFactor"].squeeze())
        manual_rotation = mat["ManualRotation"].squeeze()
        T = mat["RotMatrix"]

        theta = float(manual_rotation[0])
        dx = float(manual_rotation[1])
        dy = float(manual_rotation[2])

        print(f"  scaleFactor = {scale_factor:.3f}")
        print(f"  ManualRotation = [theta={theta}, dx={dx}, dy={dy}]")

        for reg_name in register:

            bw_path = dir_digpat / f"{name} {reg_name}.tif"
            bw = cv2.imread(str(bw_path), cv2.IMREAD_GRAYSCALE)

            bw_mask = bw > 0

            if scale_factor != 1.0: bw_resized = cv2.resize(bw_mask.astype(np.uint8), None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST,
                ) > 0
            else:
                bw_resized = bw_mask

            bw_rot = apply_manual_rotation(bw_resized, fixed_shape=(fixed_h, fixed_w), theta_deg=theta, dx=dx, dy=dy)

            bw_warped = apply_affine(bw_rot, T, fixed_shape=(fixed_h, fixed_w))

            save_path = dir_save / f"{name} {reg_name}.tif"
            bw_out = (bw_warped > 0).astype(np.uint8) * 255
            cv2.imwrite(str(save_path), bw_out)
            print(f"       Lagret: {save_path}")


if __name__ == "__main__":
    main()