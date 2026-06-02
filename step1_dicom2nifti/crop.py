#!/usr/bin/env python3
"""
Crop NIfTI phases to a cube centered at the volume center.
"""
import os
import json
import time
import numpy as np
import nibabel as nib


def _compute_center_crop(original_shape, cube_size):
    center = np.array(original_shape) // 2
    half   = cube_size // 2
    origin = np.zeros(3, dtype=int)

    for d in range(3):
        lo = center[d] - half
        hi = lo + cube_size
        if lo < 0:
            lo = 0; hi = cube_size
        if hi > original_shape[d]:
            hi = original_shape[d]; lo = hi - cube_size
        if lo < 0:
            lo = 0; hi = original_shape[d]
        origin[d] = lo

    crop_slices, actual_size = [], []
    for d in range(3):
        lo = int(origin[d])
        hi = min(lo + cube_size, original_shape[d])
        crop_slices.append(slice(lo, hi))
        actual_size.append(hi - lo)

    return {
        'crop_slices':    tuple(crop_slices),
        'crop_origin':    tuple(int(x) for x in origin),
        'crop_size':      tuple(actual_size),
        'cube_size':      cube_size,
        'original_shape': tuple(int(x) for x in original_shape),
        'center':         tuple(int(x) for x in center),
    }


def _crop_and_save(input_path, output_path, crop_info, affine_full):
    img       = nib.load(input_path)
    sl        = crop_info['crop_slices']
    data_crop = img.get_fdata()[sl[0], sl[1], sl[2]].copy()
    crop_origin = np.array(crop_info['crop_origin'], dtype=float)
    crop_affine = affine_full.copy()
    crop_affine[:3, 3] = affine_full[:3, 3] + affine_full[:3, :3] @ crop_origin
    nib.save(nib.Nifti1Image(data_crop, crop_affine), output_path)
    return data_crop.shape


def crop_phases(input_dir, output_dir, n_phases=25, cube_size=480):
    """Crop all phase NIfTI files to a cube centered at the volume center.

    Args:
        input_dir  : folder containing phase1.nii.gz ... phase{n}.nii.gz
        output_dir : where to write cropped files (same filenames)
        n_phases   : number of cardiac phases
        cube_size  : side length of the output cube (default 480)

    Also writes crop_info.json to output_dir for coordinate conversion.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[crop_phases] {input_dir} -> {output_dir}  cube={cube_size}")

    ref_img        = nib.load(os.path.join(input_dir, "phase1.nii.gz"))
    original_shape = ref_img.shape[:3]
    affine_full    = ref_img.affine
    crop_info      = _compute_center_crop(original_shape, cube_size)

    print(f"  Original shape : {original_shape}")
    print(f"  Crop origin    : {crop_info['crop_origin']}")
    print(f"  Crop size      : {crop_info['crop_size']}")

    json_path = os.path.join(output_dir, "crop_info.json")
    with open(json_path, 'w') as f:
        json.dump({k: v for k, v in crop_info.items() if k != 'crop_slices'}, f, indent=2)

    for p in range(1, n_phases + 1):
        in_path  = os.path.join(input_dir,  f"phase{p}.nii.gz")
        out_path = os.path.join(output_dir, f"phase{p}.nii.gz")
        if not os.path.exists(in_path):
            print(f"  Phase {p:2d}: SKIPPED")
            continue
        t0    = time.time()
        shape = _crop_and_save(in_path, out_path, crop_info, affine_full)
        mb    = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Phase {p:2d}: {shape}  {mb:.1f} MB  ({time.time()-t0:.1f}s)")

    # Also crop b0 if present
    b0_in  = os.path.join(input_dir,  "b0.nii.gz")
    b0_out = os.path.join(output_dir, "b0.nii.gz")
    if os.path.exists(b0_in):
        shape = _crop_and_save(b0_in, b0_out, crop_info, affine_full)
        print(f"  b0        : {shape}")

    print("  Done.")