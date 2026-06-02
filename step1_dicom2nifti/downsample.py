#!/usr/bin/env python3
"""
Downsample NIfTI phases from 960^3 to 480^3.
"""
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def downsample_phases(input_dir, output_dir, n_phases=25, order=3):
    """Downsample all phase NIfTI files from 960^3 to 480^3.

    Args:
        input_dir  : folder containing phase1.nii.gz ... phase{n}.nii.gz
        output_dir : where to write downsampled files (same filenames)
        n_phases   : number of cardiac phases
        order      : interpolation order (3=spline for intensity, 0 for labels)
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[downsample_phases] {input_dir} -> {output_dir}")

    for p in range(1, n_phases + 1):
        in_path  = os.path.join(input_dir,  f"phase{p}.nii.gz")
        out_path = os.path.join(output_dir, f"phase{p}.nii.gz")

        if not os.path.exists(in_path):
            print(f"  Phase {p:2d}: SKIPPED (not found)")
            continue

        img    = nib.load(in_path)
        data   = img.get_fdata(dtype=np.float32)
        affine = img.affine.copy()
        header = img.header.copy()

        target_shape = tuple(s // 2 for s in data.shape)
        zoom_factors = tuple(t / s for t, s in zip(target_shape, data.shape))

        print(f"  Phase {p:2d}: {data.shape} -> {target_shape} ... ", end="", flush=True)
        data_ds = zoom(data, zoom_factors, order=order, prefilter=True)

        new_affine = affine.copy()
        new_affine[:3, :3] = affine[:3, :3] / np.array(zoom_factors)
        old_vox = np.abs(np.diag(affine)[:3])
        new_vox = old_vox / np.array(zoom_factors)
        shift   = (new_vox - old_vox) / 2.0
        for i in range(3):
            new_affine[:3, 3] += affine[:3, i] / np.linalg.norm(affine[:3, i]) * shift[i]

        out_img = nib.Nifti1Image(data_ds, new_affine, header)
        out_img.header.set_data_shape(data_ds.shape)
        out_img.header.set_zooms(np.abs(np.diag(new_affine))[:3])
        nib.save(out_img, out_path)
        print("OK")

    # Also downsample b0 if present
    for fname in ["b0.nii.gz"]:
        in_path  = os.path.join(input_dir,  fname)
        out_path = os.path.join(output_dir, fname)
        if not os.path.exists(in_path):
            continue
        img    = nib.load(in_path)
        data   = img.get_fdata(dtype=np.float32)
        affine = img.affine.copy()
        header = img.header.copy()
        target_shape = tuple(s // 2 for s in data.shape)
        zoom_factors = tuple(t / s for t, s in zip(target_shape, data.shape))
        print(f"  b0        : {data.shape} -> {target_shape} ... ", end="", flush=True)
        data_ds = zoom(data, zoom_factors, order=order, prefilter=True)
        new_affine = affine.copy()
        new_affine[:3, :3] = affine[:3, :3] / np.array(zoom_factors)
        old_vox = np.abs(np.diag(affine)[:3])
        new_vox = old_vox / np.array(zoom_factors)
        shift   = (new_vox - old_vox) / 2.0
        for i in range(3):
            new_affine[:3, 3] += affine[:3, i] / np.linalg.norm(affine[:3, i]) * shift[i]
        out_img = nib.Nifti1Image(data_ds, new_affine, header)
        out_img.header.set_data_shape(data_ds.shape)
        out_img.header.set_zooms(np.abs(np.diag(new_affine))[:3])
        nib.save(out_img, out_path)
        print("OK")

    print("  Done.")