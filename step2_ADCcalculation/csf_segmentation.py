#!/usr/bin/env python3
"""
CSF segmentation: apply thresholds to FSL FAST output and brain mask.
"""

import os
import struct
import gzip
import numpy as np


def _get_dtype_info(datatype):
    return {
        2: (np.uint8, 1), 4: (np.int16, 2), 8: (np.int32, 4),
        16: (np.float32, 4), 64: (np.float64, 8),
        512: (np.uint16, 2), 768: (np.uint32, 4),
    }.get(datatype, (np.float32, 4))


def _read_nifti(filepath):
    with gzip.open(filepath, 'rb') as f:
        header_data    = f.read(348)
        extension_bytes = f.read(4)
        data_bytes     = f.read()
    datatype = struct.unpack('<h', header_data[70:72])[0]
    dtype, bpv = _get_dtype_info(datatype)
    n = len(data_bytes) // bpv
    dim = round(n ** (1/3))
    data = np.frombuffer(data_bytes, dtype=dtype)[:dim**3].reshape((dim, dim, dim))
    return header_data, extension_bytes, data, dtype


def _write_nifti(filepath, header, ext, data, dtype):
    with gzip.open(filepath, 'wb', compresslevel=6) as f:
        f.write(header)
        f.write(ext)
        f.write(data.astype(dtype).tobytes())
    print(f"  Saved: {os.path.basename(filepath)}")


def csf_segmentation(skull_less_brain_path, csf_seg_path, output_dir, subject_id,
                     csf_thresholds=None, brain_threshold_percent=0.1):
    """Apply thresholds to skull-stripped brain and FSL FAST CSF segmentation.

    Args:
        skull_less_brain_path    : path to skull-stripped brain .nii.gz (960^3)
        csf_seg_path             : path to FSL FAST pve_0 upsampled to 960 .nii.gz
        output_dir               : where to write outputs
        subject_id               : used in output filenames (e.g. '250701')
        csf_thresholds           : list of CSF probability thresholds (default [0.6])
        brain_threshold_percent  : brain mask threshold as fraction of max (default 0.1)

    Outputs (in output_dir):
        matrix2_threshold_10percent_{subject_id}.nii.gz   brain mask
        csfseg_threshold_{subject_id}_{thresh}.nii.gz     CSF mask per threshold
        interseg_threshold_{subject_id}_{thresh}.nii.gz   intersection per threshold
    """
    if csf_thresholds is None:
        csf_thresholds = [0.6]

    os.makedirs(output_dir, exist_ok=True)
    print(f"[csf_segmentation] subject={subject_id}")

    brain_header, brain_ext, brain_data, brain_dtype = _read_nifti(skull_less_brain_path)
    csf_header,   csf_ext,   csf_data,   csf_dtype   = _read_nifti(csf_seg_path)

    if brain_data.shape != csf_data.shape:
        raise ValueError(f"Shape mismatch: brain={brain_data.shape}, CSF={csf_data.shape}")

    print(f"  Volume shape : {brain_data.shape}")

    # Brain mask
    thr = brain_data.max() * brain_threshold_percent
    brain_mask = (brain_data >= thr).astype(np.uint8)
    print(f"  Brain voxels : {brain_mask.sum()}")
    _write_nifti(
        os.path.join(output_dir, f"matrix2_threshold_10percent_{subject_id}.nii.gz"),
        brain_header, brain_ext, brain_mask, brain_dtype
    )

    # CSF thresholds
    for thresh in csf_thresholds:
        csf_mask  = (csf_data >= thresh).astype(np.uint8)
        inter_mask = csf_mask & brain_mask
        print(f"  thresh={thresh}: CSF={csf_mask.sum()}, intersection={inter_mask.sum()}")

        _write_nifti(
            os.path.join(output_dir, f"csfseg_threshold_{subject_id}_{thresh:.1f}.nii.gz"),
            csf_header, csf_ext, csf_mask, csf_dtype
        )
        _write_nifti(
            os.path.join(output_dir, f"interseg_threshold_{subject_id}_{thresh:.1f}.nii.gz"),
            csf_header, csf_ext, inter_mask, csf_dtype
        )

    print("  Done.")