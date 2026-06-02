#!/usr/bin/env python3
"""
ADC calculation and correction.
Supports two input modes:
  mode='dicom'  - b0 and b40 loaded from DICOM folders
  mode='nifti'  - b0 and b40 loaded from NIfTI files, usually for scan 2 that only has nifti images
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hdf5storage


# ── helpers ──────────────────────────────────────────────────────────────────

def _save_slice(data, filepath, title="", slice_idx=None):
    if slice_idx is None:
        slice_idx = data.shape[2] // 2
    plt.figure(figsize=(10, 10))
    plt.imshow(data[:, :, slice_idx], cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  QC image: {os.path.basename(filepath)}")


def _save_mat(filepath, data_dict):
    if os.path.exists(filepath):
        os.remove(filepath)
    hdf5storage.savemat(filepath, data_dict, format='7.3',
                        oned_as='column', store_python_metadata=False)
    print(f"  Saved: {os.path.basename(filepath)}")


def _load_nifti(filepath):
    import nibabel as nib
    print(f"  Loading NIfTI: {os.path.basename(filepath)}")
    data = np.asarray(nib.load(filepath).dataobj, dtype=np.float32)
    data = np.transpose(data, (1, 0, 2))   # MATLAB permute([2,1,3])
    print(f"  Shape: {data.shape}")
    return data


def _load_dicom_volume(folder_path, image_size):
    import pydicom
    print(f"  Loading DICOM: {os.path.basename(folder_path)}")
    files = sorted([f for f in os.listdir(folder_path)
                    if f.startswith('img_') and f.endswith('.dcm')])
    if not files:
        raise ValueError(f"No DICOM files in {folder_path}")
    volume = np.zeros(image_size, dtype=np.float32)
    for fname in files:
        sl = int(fname.replace('img_', '').replace('.dcm', ''))
        if sl < image_size[2]:
            dcm = pydicom.dcmread(os.path.join(folder_path, fname))
            volume[:, :, sl] = dcm.pixel_array.astype(np.float32)
    print(f"  Loaded {len(files)} slices")
    return volume


def _load_csf_mask(filepath):
    print(f"  Loading CSF mask: {os.path.basename(filepath)}")
    try:
        import h5py
        with h5py.File(filepath, 'r') as f:
            keys = [k for k in f.keys() if not k.startswith('#')]
            key  = 'data' if 'data' in f else ('csfMask' if 'csfMask' in f else keys[0])
            mask = np.array(f[key])
            if mask.ndim == 3:
                mask = np.transpose(mask, (2, 1, 0))
    except (ImportError, OSError):
        from scipy.io import loadmat
        mat  = loadmat(filepath)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        key  = 'data' if 'data' in mat else ('csfMask' if 'csfMask' in mat else keys[0])
        mask = mat[key]
    mask = np.transpose(mask, (1, 0, 2))   # match original code convention
    print(f"  Shape: {mask.shape}, nonzero: {np.count_nonzero(mask)}")
    return mask


def _compute_correction_constants(d, TEp, TR, segment, TE, fs, alpha, b):
    echo_spacing = 2 * TE
    trc  = TR - (4 + segment) * echo_spacing - fs - TE
    T1   = 4000.0
    T2   = 2000.0
    E1   = np.exp(-echo_spacing / T1)
    E2   = np.exp(-echo_spacing / T2)
    Oz   = ((1 - E1) * (E2 + np.cos(alpha))) / \
           ((1 - E1*np.cos(alpha)) - (E1 - np.cos(alpha))*E2)
    Op   = np.exp(-TEp / T2) * np.exp(-d / T1)
    OB   = Op * Oz * np.exp(-trc / T1)
    Od   = 1 - np.exp(-trc / T1)
    Orc  = Od
    OA   = Orc + Od * Oz * np.exp(-trc / T1)
    print(f"  E1={E1:.6f}, E2={E2:.6f}, Oz={Oz:.6f}, OB={OB:.6f}")
    return OA, OB, Od, Op, b


# ── main function ─────────────────────────────────────────────────────────────

def calculate_adc(
    output_dir,
    csf_mask_path,
    mode,                          # 'nifti' or 'dicom'
    # NIfTI mode
    b0_nifti_path=None,
    b40_nifti_dir=None,
    b40_nifti_template="r_phase{phase}.nii.gz",
    # DICOM mode
    b0_dicom_dir=None,
    b40_dicom_base_dir=None,
    image_size=(960, 960, 960),
    # Sequence parameters
    d=22, TEp=52, TR=240, segment=32, TE=2.79, fs=16,
    alpha=None, b=40,
    n_phases=25,
):
    """Calculate and correct ADC maps for all cardiac phases.

    Args:
        output_dir          : where to write .mat outputs and QC images
        csf_mask_path       : path to csfmask.mat
        mode                : 'nifti' or 'dicom'
        b0_nifti_path       : (nifti mode) path to b0.nii.gz
        b40_nifti_dir       : (nifti mode) folder containing phase NIfTIs
        b40_nifti_template  : (nifti mode) filename template, e.g. 'r_phase{phase}.nii.gz'
        b0_dicom_dir        : (dicom mode) folder with b0 DICOM files
        b40_dicom_base_dir  : (dicom mode) base folder containing dicom_bbcine_phase{n}/ subfolders
        image_size          : (dicom mode) expected volume size (default 960^3)
        d, TEp, TR, segment, TE, fs : sequence parameters (ms)
        alpha               : flip angle in radians (default pi/4)
        b                   : b-value (s/mm2, default 40)
        n_phases            : number of cardiac phases
    """
    if alpha is None:
        alpha = np.pi / 4

    qc_dir = os.path.join(output_dir, "qc_images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    print("=" * 70)
    print(f"ADC Calculation  mode={mode}")
    print("=" * 70)

    # Load CSF mask
    print("\nLoading CSF mask...")
    csf_mask = _load_csf_mask(csf_mask_path)

    # Load b0
    print("\nLoading b=0...")
    if mode == 'nifti':
        b0 = _load_nifti(b0_nifti_path)
    else:
        b0 = _load_dicom_volume(b0_dicom_dir, image_size)
    _save_slice(b0, os.path.join(qc_dir, 'b0_image.png'), 'b=0')

    # Correction constants
    print("\nComputing correction constants...")
    OA, OB, Od, Op, b_val = _compute_correction_constants(
        d, TEp, TR, segment, TE, fs, alpha, b)

    # Per-phase loop
    for phase in range(1, n_phases + 1):
        print(f"\n{'='*70}\nPhase {phase}/{n_phases}")

        # Load b40
        if mode == 'nifti':
            b40_path = os.path.join(b40_nifti_dir,
                                    b40_nifti_template.format(phase=phase))
            if not os.path.exists(b40_path):
                print(f"  SKIPPED (not found): {b40_path}")
                continue
            b40 = _load_nifti(b40_path)
        else:
            phase_dir = os.path.join(b40_dicom_base_dir, f"dicom_bbcine_phase{phase}")
            if not os.path.exists(phase_dir):
                print(f"  SKIPPED (not found): {phase_dir}")
                continue
            b40 = _load_dicom_volume(phase_dir, image_size)

        if b40.shape != b0.shape:
            raise ValueError(f"Phase {phase}: shape mismatch b40={b40.shape} b0={b0.shape}")

        if phase == 1:
            _save_slice(b40, os.path.join(qc_dir, 'b40_phase01.png'), f'b={b} Phase 1')

        # Raw ADC
        eps = 1e-10
        adc = -np.log(np.maximum(b40, eps) / np.maximum(b0, eps)) / b_val
        if phase == 1:
            _save_slice(adc, os.path.join(qc_dir, 'adc_raw_phase01.png'), 'ADC raw Phase 1')
        _save_mat(os.path.join(output_dir, f'adc_phase{phase:02d}_single.mat'),
                  {'adcImage': adc})

        # Corrected ADC
        valid = (adc > 0) & np.isfinite(adc)
        adc_corr = np.zeros_like(adc)
        if valid.sum() > 0:
            af     = adc[valid]
            E      = np.exp(af * b_val)
            Ok     = (((Od + (OA * Op / (1 - OB))) / E) - Od) / (OA * Op)
            G      = np.log(1 + Ok * OB) - np.log(Ok)
            adc_corr[valid] = G / b_val
        adc_corr[~np.isfinite(adc_corr)] = 0
        adc_corr[adc_corr < 0] = 0
        print(f"  Corrected ADC range: {adc_corr.min():.6f} - {adc_corr.max():.6f}")

        if phase == 1:
            _save_slice(adc_corr, os.path.join(qc_dir, 'adc_corrected_phase01.png'),
                        'ADC corrected Phase 1')
        _save_mat(os.path.join(output_dir, f'adc_corrected_phase{phase:02d}_single.mat'),
                  {'adcCorrected': adc_corr})

        # CSF masked
        adc_csf = adc_corr * csf_mask.astype(np.float32)
        _save_slice(adc_csf,
                    os.path.join(qc_dir, f'adc_corrected_csf_phase{phase:02d}.png'),
                    f'ADC corrected + CSF mask Phase {phase}')
        _save_mat(os.path.join(output_dir, f'adc_corrected_csf_phase{phase:02d}_single.mat'),
                  {'adcCorrectedCSF': adc_csf})

        del b40, adc, adc_corr, adc_csf

    print("\n" + "=" * 70)
    print("ADC calculation complete.")
    print(f"  Output : {output_dir}")
    print(f"  QC     : {qc_dir}")
    print("=" * 70)