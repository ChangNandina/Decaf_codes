#!/usr/bin/env python3
"""
Step 1: DICOM -> NIfTI pipeline
================================
for new subject, change CONFIG, then run:
    python run_step1.py

Steps:
  1a. DICOM -> NIfTI (960)
  1b. downsample 960 -> 480       (for fsl fast CSF input, if only want vessel result, ignore this)
  1c. Crop center 480^3 cube      (for itksnap labeling, need install itksnap nninteractive environment)
  1d. register scan2 to scan1     (for 2 scans reproducibility, if only one scan, ignore this)
"""

from convert_dicom_to_nifti import convert_dicom_to_nifti
from downsample import downsample_phases
from crop import crop_phases
from register import register_interscan

import os

# ============================================================
# CONFIG
# ============================================================

SUBJECT_ID  = "250701"                        # 用于命名输出文件夹
BASE_OUT    = "/v/ai/nobackup/cni/chang_data" # 输出根目录
N_PHASES    = 25

# --- Scan 1 ---
B40_DICOM_DIR_S1 = "/v/ai/nobackup/xma/Trufi_BBCine_results/Dicom_New_960_all/20250701_human_dwtrufi_old2/meas_MID00033_FID136887_trufi_b40_Kooshball_Nphs25_NCha32"
B0_DICOM_DIR_S1  = None      # 填路径或保持 None
B0_SUBDIR_S1     = "dicom_bbcine_combined"

# --- Scan 2 (第二次扫描，没有就保持 None) ---
B40_DICOM_DIR_S2 = None
B0_DICOM_DIR_S2  = None
B0_SUBDIR_S2     = "dicom_bbcine_combined"

# --- 选择要跑哪些步骤 ---
RUN_DICOM2NIFTI = True
RUN_DOWNSAMPLE  = True
RUN_CROP        = True
RUN_REGISTER    = False   # 只有 S2 不为 None 时才有效


# ============================================================

nifti_dir_s1 = os.path.join(BASE_OUT, f"{SUBJECT_ID}_960", "nifti_960_s1")
nifti_dir_s2 = os.path.join(BASE_OUT, f"{SUBJECT_ID}_960", "nifti_960_s2")
ds_dir_s1    = os.path.join(BASE_OUT, f"{SUBJECT_ID}_960", "nifti_480_s1")
ds_dir_s2    = os.path.join(BASE_OUT, f"{SUBJECT_ID}_960", "nifti_480_s2")
crop_dir_s1  = os.path.join(BASE_OUT, f"{SUBJECT_ID}_960", "crop480_s1")
crop_dir_s2  = os.path.join(BASE_OUT, f"{SUBJECT_ID}_960", "crop480_s2")

print("=" * 60)
print(f"Subject: {SUBJECT_ID}")
print("=" * 60)

# Step 1a: DICOM -> NIfTI
if RUN_DICOM2NIFTI:
    print("\n[Step 1a] DICOM -> NIfTI  (Scan 1)")
    convert_dicom_to_nifti(B40_DICOM_DIR_S1, nifti_dir_s1, N_PHASES,
                           b0_dicom_dir=B0_DICOM_DIR_S1, b0_subdir=B0_SUBDIR_S1)
    if B40_DICOM_DIR_S2:
        print("\n[Step 1a] DICOM -> NIfTI  (Scan 2)")
        convert_dicom_to_nifti(B40_DICOM_DIR_S2, nifti_dir_s2, N_PHASES,
                               b0_dicom_dir=B0_DICOM_DIR_S2, b0_subdir=B0_SUBDIR_S2)

# Step 1b: Downsample 960 -> 480
if RUN_DOWNSAMPLE:
    print("\n[Step 1b] Downsample 960 -> 480  (Scan 1)")
    downsample_phases(nifti_dir_s1, ds_dir_s1, N_PHASES)
    if B40_DICOM_DIR_S2:
        print("\n[Step 1b] Downsample 960 -> 480  (Scan 2)")
        downsample_phases(nifti_dir_s2, ds_dir_s2, N_PHASES)

# Step 1c: Crop center 480^3
if RUN_CROP:
    print("\n[Step 1c] Crop center 480  (Scan 1)")
    crop_phases(nifti_dir_s1, crop_dir_s1, N_PHASES, cube_size=480)
    if B40_DICOM_DIR_S2:
        print("\n[Step 1c] Crop center 480  (Scan 2)")
        crop_phases(nifti_dir_s2, crop_dir_s2, N_PHASES, cube_size=480)

# Step 1d: Register scan2 -> scan1
if RUN_REGISTER and B40_DICOM_DIR_S2:
    print("\n[Step 1d] Register Scan 2 -> Scan 1")
    register_interscan(ref_dir=nifti_dir_s1, src_dir=nifti_dir_s2)


print("\n" + "=" * 60)
print("Step 1 complete.")
print("=" * 60)