#!/usr/bin/env python3
"""
Step 2: ADC Calculation Pipeline
==================================
For a new subject, change CONFIG then run:
    python run_step2.py

Steps:
  2a. Upsample FSL FAST output 480 -> 960
  2b. CSF segmentation (brain mask + CSF threshold)
  2c. Morphological opening -> csfmask.nii.gz + QC overlay PNG
      [Review QC image, then set RUN_ADC=True and re-run]
  2d. MATLAB: convert csfmask.nii.gz -> csfmask.mat
  2e. ADC calculation and correction
"""

from upsample import upsample_to_960
from csf_segmentation import csf_segmentation
from csf_mask_utils import generate_csf_mask, generate_overlay_qc, save_nifti_to_mat
from adc_calculation import calculate_adc
import os

# ============================================================
# CONFIG — only change this section for each subject
# ============================================================

SUBJECT_ID = "250701"
BASE_DIR   = f"/v/ai/nobackup/cni/chang_data/{SUBJECT_ID}_960"
N_PHASES   = 25

# --- Step 2a: Upsample FSL FAST outputs ---
FSL_PVE0_480_PATH  = os.path.join(BASE_DIR, "phase1_480_brain_pve_0.nii.gz")
FSL_PVE0_960_PATH  = os.path.join(BASE_DIR, "phase1_brain_pve_960.nii.gz")
FSL_BRAIN_480_PATH = os.path.join(BASE_DIR, "phase1_480_brain.nii.gz")
FSL_BRAIN_960_PATH = os.path.join(BASE_DIR, "phase1_brain.nii.gz")
UPSAMPLE_METHOD    = "nearest"

# --- Step 2b: CSF segmentation ---
CSF_OUTPUT_DIR    = BASE_DIR
CSF_THRESHOLDS    = [0.6]
BRAIN_THR_PERCENT = 0.1

# --- Step 2c: Morphological opening + QC ---
INTERSEG_PATH     = os.path.join(BASE_DIR, f"interseg_threshold_{SUBJECT_ID}_0.6.nii.gz")
CSFMASK_NII_PATH  = os.path.join(BASE_DIR, f"csfmask_{SUBJECT_ID}.nii.gz")
QC_PNG_PATH       = os.path.join(BASE_DIR, "qc", f"csfmask_overlay_{SUBJECT_ID}.png")
OPENING_ITER      = 1   # erosion + dilation iterations

# --- Step 2d+2e: ADC ---
CSFMASK_MAT_PATH  = os.path.join(BASE_DIR, f"csfmask_{SUBJECT_ID}.mat")
ADC_OUTPUT_DIR    = os.path.join(BASE_DIR, f"adc_corrected_maps_{SUBJECT_ID}")
ADC_MODE          = "dicom"   # 'dicom' or 'nifti'

# DICOM mode
B0_DICOM_DIR       = "/v/ai/nobackup/xma/Trufi_BBCine_results/Dicom_New_960_all/20250701_human_dwtrufi_old2/meas_MID00033_FID136887_trufi_b40_Kooshball_Nphs25_NCha32_retrogate_b0_cgsense/dicom_bbcine_combined"
B40_DICOM_BASE_DIR = "/v/ai/nobackup/xma/Trufi_BBCine_results/Dicom_New_960_all/20250701_human_dwtrufi_old2/meas_MID00033_FID136887_trufi_b40_Kooshball_Nphs25_NCha32_retrogate_l1_ttv_grasp_kwic_Nrings5_TV0.001_TTV0.01"

# NIfTI mode (used when ADC_MODE = 'nifti')
B0_NIFTI_PATH      = os.path.join(BASE_DIR, "nifti_960_s2_sitk_rigid", "r_b0.nii.gz")
B40_NIFTI_DIR      = os.path.join(BASE_DIR, "nifti_960_s2_sitk_rigid")
B40_NIFTI_TEMPLATE = "r_phase{phase}.nii.gz"

ADC_PARAMS = dict(d=22, TEp=52, TR=240, segment=32, TE=2.79, fs=16, b=40)

# --- Select steps ---
RUN_UPSAMPLE = True
RUN_CSF_SEG  = True
RUN_CSF_MASK = True   # morphological opening + QC image
RUN_ADC      = False  # set True after reviewing QC image

# ============================================================

print("=" * 60)
print(f"Step 2  |  Subject: {SUBJECT_ID}")
print("=" * 60)

# Step 2a
if RUN_UPSAMPLE:
    print("\n[Step 2a] Upsample FSL outputs 480 -> 960")
    upsample_to_960(FSL_PVE0_480_PATH,  FSL_PVE0_960_PATH,  method=UPSAMPLE_METHOD)
    upsample_to_960(FSL_BRAIN_480_PATH, FSL_BRAIN_960_PATH, method=UPSAMPLE_METHOD)

# Step 2b
if RUN_CSF_SEG:
    print("\n[Step 2b] CSF segmentation")
    csf_segmentation(
        skull_less_brain_path   = FSL_BRAIN_960_PATH,
        csf_seg_path            = FSL_PVE0_960_PATH,
        output_dir              = CSF_OUTPUT_DIR,
        subject_id              = SUBJECT_ID,
        csf_thresholds          = CSF_THRESHOLDS,
        brain_threshold_percent = BRAIN_THR_PERCENT,
    )

# Step 2c
if RUN_CSF_MASK:
    print("\n[Step 2c] Morphological opening -> csfmask + QC overlay")
    generate_csf_mask(INTERSEG_PATH, CSFMASK_NII_PATH, opening_iterations=OPENING_ITER)
    generate_overlay_qc(FSL_BRAIN_960_PATH, CSFMASK_NII_PATH, QC_PNG_PATH)
    print(f"\n{'='*60}")
    print(f"  Review QC image: {QC_PNG_PATH}")
    print(f"  If mask looks OK: set RUN_UPSAMPLE=False, RUN_CSF_SEG=False,")
    print(f"                        RUN_CSF_MASK=False, RUN_ADC=True")
    print(f"  Then re-run: python run_step2.py")
    print(f"{'='*60}")

# Step 2d + 2e
if RUN_ADC:
    print("\n[Step 2d] MATLAB: csfmask.nii.gz -> csfmask.mat")
    save_nifti_to_mat(CSFMASK_NII_PATH, CSFMASK_MAT_PATH)

    print("\n[Step 2e] ADC calculation and correction")
    if ADC_MODE == 'dicom':
        calculate_adc(
            output_dir         = ADC_OUTPUT_DIR,
            csf_mask_path      = CSFMASK_MAT_PATH,
            mode               = 'dicom',
            b0_dicom_dir       = B0_DICOM_DIR,
            b40_dicom_base_dir = B40_DICOM_BASE_DIR,
            n_phases           = N_PHASES,
            **ADC_PARAMS,
        )
    else:
        calculate_adc(
            output_dir          = ADC_OUTPUT_DIR,
            csf_mask_path       = CSFMASK_MAT_PATH,
            mode                = 'nifti',
            b0_nifti_path       = B0_NIFTI_PATH,
            b40_nifti_dir       = B40_NIFTI_DIR,
            b40_nifti_template  = B40_NIFTI_TEMPLATE,
            n_phases            = N_PHASES,
            **ADC_PARAMS,
        )

print("\n" + "=" * 60)
print("Step 2 complete.")
print("=" * 60)