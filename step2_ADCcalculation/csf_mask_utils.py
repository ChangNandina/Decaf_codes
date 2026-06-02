#!/usr/bin/env python3
"""
Generate a three-plane QC overlay image: brain underlay + CSF mask overlay.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation


def apply_morphological_opening(mask, iterations=1):
    """Erosion followed by dilation to remove isolated small mask points."""
    struct = np.ones((3, 3, 3), dtype=bool)
    eroded  = binary_erosion(mask, structure=struct, iterations=iterations)
    dilated = binary_dilation(eroded, structure=struct, iterations=iterations)
    return dilated.astype(np.uint8)


def generate_csf_mask(interseg_path, output_nifti_path, opening_iterations=1):
    """Apply morphological opening to interseg and save as csfmask.nii.gz

    Args:
        interseg_path       : path to interseg_threshold_*.nii.gz
        output_nifti_path   : where to save csfmask.nii.gz
        opening_iterations  : erosion/dilation radius in voxels (default 1)
    """
    print(f"[generate_csf_mask]")
    print(f"  Input  : {os.path.basename(interseg_path)}")

    img  = nib.load(interseg_path)
    data = np.asarray(img.dataobj).astype(bool)
    print(f"  Shape  : {data.shape}, nonzero before: {data.sum()}")

    cleaned = apply_morphological_opening(data, iterations=opening_iterations)
    removed = data.sum() - cleaned.sum()
    print(f"  Removed: {removed} voxels after opening (iter={opening_iterations})")
    print(f"  Nonzero after: {cleaned.sum()}")

    os.makedirs(os.path.dirname(output_nifti_path), exist_ok=True)
    nib.save(nib.Nifti1Image(cleaned, img.affine, img.header), output_nifti_path)
    print(f"  Saved  : {os.path.basename(output_nifti_path)}")
    return cleaned


def generate_overlay_qc(brain_path, mask_path, output_png_path,
                         n_slices=7, alpha=0.4):
    """Generate a three-plane QC figure: brain + CSF mask overlay.

    Args:
        brain_path      : path to skull-stripped brain .nii.gz (960^3)
        mask_path       : path to csfmask .nii.gz
        output_png_path : where to save the QC PNG
        n_slices        : number of evenly-spaced slices per plane (default 7)
        alpha           : mask overlay transparency (default 0.4)
    """
    print(f"[generate_overlay_qc]")

    brain = np.asarray(nib.load(brain_path).dataobj, dtype=np.float32)
    mask  = np.asarray(nib.load(mask_path).dataobj).astype(bool)

    print(f"  Brain shape: {brain.shape}, Mask nonzero: {mask.sum()}")

    # Normalize brain for display
    p2, p98 = np.percentile(brain[brain > 0], [2, 98])
    brain_norm = np.clip((brain - p2) / (p98 - p2 + 1e-8), 0, 1)

    # Pick evenly spaced slices around the center for each plane
    def get_slices(size, n):
        margin = size // 6
        return np.linspace(margin, size - margin, n, dtype=int)

    sx = get_slices(brain.shape[0], n_slices)
    sy = get_slices(brain.shape[1], n_slices)
    sz = get_slices(brain.shape[2], n_slices)

    fig, axes = plt.subplots(3, n_slices, figsize=(n_slices * 3, 10))
    fig.patch.set_facecolor('black')

    plane_configs = [
        ("Axial (z)",    sz,  lambda i: (brain_norm[:, :, i], mask[:, :, i])),
        ("Coronal (y)",  sy,  lambda i: (brain_norm[:, i, :], mask[:, i, :])),
        ("Sagittal (x)", sx,  lambda i: (brain_norm[i, :, :], mask[i, :, :])),
    ]

    for row, (plane_label, slices, get_data) in enumerate(plane_configs):
        for col, sl in enumerate(slices):
            ax = axes[row, col]
            b_slice, m_slice = get_data(sl)

            # Brain underlay
            ax.imshow(np.rot90(b_slice), cmap='gray', vmin=0, vmax=1)

            # Mask overlay in cyan
            overlay = np.zeros((*b_slice.shape, 4))
            overlay[m_slice] = [0, 1, 1, alpha]   # RGBA cyan
            ax.imshow(np.rot90(overlay))

            ax.axis('off')
            if col == 0:
                ax.set_ylabel(plane_label, color='white', fontsize=10)
            if row == 0:
                ax.set_title(f"sl={sl}", color='white', fontsize=8)

    fig.suptitle("CSF Mask QC — cyan overlay on brain\n"
                 f"(brain: {os.path.basename(brain_path)}  "
                 f"mask: {os.path.basename(mask_path)})",
                 color='white', fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=150, bbox_inches='tight',
                facecolor='black')
    plt.close()
    print(f"  QC image saved: {output_png_path}")


def save_nifti_to_mat(nifti_path, mat_path):
    """Call MATLAB (server) to convert csfmask.nii.gz to .mat (v7.3)."""
    import subprocess
    matlab_cmd = (
        f"data = niftiread('{nifti_path}'); "
        f"save('{mat_path}', 'data', '-v7.3');"
    )
    print(f"[save_nifti_to_mat] Running MATLAB...")
    result = subprocess.run(
        ["matlab", "-nodisplay", "-nosplash", "-r", f"{matlab_cmd} exit"],
        capture_output=True, text=True, timeout=180
    )
    if result.returncode != 0:
        print(f"  MATLAB stderr:\n{result.stderr}")
        raise RuntimeError("MATLAB conversion failed")
    print(f"  Saved: {os.path.basename(mat_path)}")