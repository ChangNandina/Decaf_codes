#!/usr/bin/env python3
"""
Step 3: Vessel segmentation + analysis pipeline
================================================
For a new subject, change CONFIG, then run in stages:

  Stage A (before manual review):
    python run_step3.py   with RUN_PART_A=True, others=False
    → outputs: skeleton visualizations + phase 1 segmentation

  [Manual: open ITK-SNAP, check skeleton_segments.nii.gz + p01_multilabel.nii.gz,
           set SKIP_SEGMENTS, confirm nnInteractive green light]

  Stage B (after review):
    python run_step3.py   with RUN_PART_B=True, others=False

  Stage C (analysis):
    python run_step3.py   with RUN_BUILD_GRAPH=True, RUN_AREA_PI=True,
                               RUN_EXPORT=True, others=False

Environment requirements:
  - nnInteractive annotation environment must be configured
  - ITK-SNAP must be installed and open
  - nnInteractive server must be running (green light in ITK-SNAP plugin)
  - Start server at: http://localhost:8912 before running Part A or B
"""

from planb_inference import run_part_a, run_part_b
from build_graph import build_graph
from area_pi import compute_area_pi
from export_results import export_vessel_only, export_with_csf
import os

# ============================================================
# CONFIG — only change this section for each subject
# ============================================================

SUBJECT_ID = "250701"
BASE_DIR   = f"/v/ai/nobackup/cni/chang_data/{SUBJECT_ID}_960"
N_PHASES   = 25

# Input: cropped 480^3 NIfTI phases (from step1 crop output)
NIFTI_DIR  = os.path.join(BASE_DIR, "crop480_s1")

# Phase 1 vessel mask (manually drawn in ITK-SNAP before step3)
MASK_PATH  = os.path.join(NIFTI_DIR, "p1_m.nii.gz")

# Output directories
PLANB_OUTPUT_DIR = os.path.join(BASE_DIR, f"output_{SUBJECT_ID}")
STEP1_DIR        = os.path.join(PLANB_OUTPUT_DIR, "results/step1_fromB")
STEP2_DIR        = os.path.join(PLANB_OUTPUT_DIR, "results/step2_multilabel")

# --- Part A/B: nnInteractive segmentation ---
TOTAL_POINTS      = 100
MIN_BRANCH_LENGTH = 1
BP_DILATION       = 0
MIN_SPUR_LENGTH   = 3
SOFT_MARGIN       = 5
N_WORKERS_EDT     = 16
BBOX_PAD          = 80

# SKIP_SEGMENTS: fill after reviewing Part A results in ITK-SNAP
# Check skeleton_segments.nii.gz and p01_multilabel.nii.gz
SKIP_SEGMENTS = []   # e.g. [9, 10, 11, 16]

# --- Graph building ---
BIF_CLUSTER_RADIUS = 3
EP_CLUSTER_RADIUS  = 2
MASK_DOWNSAMPLE    = 2

# --- Area / PI ---
SAMPLE_SPACING_MM   = 0.5
MIN_SAMPLES         = 5
MAX_SAMPLES         = 150
MIN_SEG_LENGTH_MM   = 3.0
S_SKIP_MM           = 2.0
SLAB_HALF_MM        = 0.5
R_MAX_MM            = 20.0
RECENTER_RADIUS_MM  = 2.5
RECENTER_SLAB_MM    = 1.0
SMOOTH_WINDOW       = 15
PATH_SMOOTH_MM      = 1.0
XSEC_N_SLICES       = 9
XSEC_RADIUS_MM      = 5.0
XSEC_RESOLUTION     = 0.2
XSEC_PHASE          = 1

# --- CSF ADC (only used when HAS_CSF=True) ---
HAS_CSF          = False   # set True if ADC maps available from step2
ADC_FOLDER       = os.path.join(BASE_DIR, f"adc_corrected_maps_{SUBJECT_ID}")
CROP_DIR         = NIFTI_DIR   # folder with crop_info.json
N_WORKERS_ADC    = 6
PERIVAS_DIST_MM  = 3.0
GRID_RES_MM      = 0.2
GRID_HALF_MM     = 6.0

# Output .mat paths
OUT_MAT_VESSEL   = os.path.join(STEP2_DIR, f"vessel_pi_{SUBJECT_ID}.mat")
OUT_MAT_COMBINED = os.path.join(STEP2_DIR, f"vessel_csf_{SUBJECT_ID}.mat")

# --- Select steps ---
RUN_PART_A      = True    # Part A: skeleton + phase1 inference
RUN_PART_B      = False   # Part B: batch all phases (after reviewing Part A)
RUN_BUILD_GRAPH = False   # Build vessel graph + 3D HTML
RUN_AREA_PI     = False   # Cross-section area + PI
RUN_EXPORT      = False   # Export .mat

# ============================================================

print("=" * 60)
print(f"Step 3  |  Subject: {SUBJECT_ID}")
print("=" * 60)

if RUN_PART_A or RUN_PART_B:
    print("""
  ┌─────────────────────────────────────────────────────┐
  │  ENVIRONMENT CHECK before running Part A or B:      │
  │  1. nnInteractive annotation environment configured  │
  │  2. ITK-SNAP is open, ssh to server                  |
  |  python -m itksnap_dls --host 0.0.0.0 --port 8912    │
  │  3. nnInteractive plugin shows GREEN light           │
  │  (server running at http://localhost:8912)  or others│
  └─────────────────────────────────────────────────────┘
""")

# Part A
if RUN_PART_A:
    print("\n[Step 3 Part A] Skeleton + seeds + phase 1 inference")
    run_part_a(
        nifti_dir         = NIFTI_DIR,
        mask_path         = MASK_PATH,
        output_dir        = PLANB_OUTPUT_DIR,
        total_points      = TOTAL_POINTS,
        min_branch_length = MIN_BRANCH_LENGTH,
        bp_dilation       = BP_DILATION,
        min_spur_length   = MIN_SPUR_LENGTH,
        soft_margin       = SOFT_MARGIN,
        n_workers         = N_WORKERS_EDT,
        bbox_pad          = BBOX_PAD,
        skip_segments     = SKIP_SEGMENTS,
    )
    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  MANUAL REVIEW in ITK-SNAP:                         │
  │  Open: {PLANB_OUTPUT_DIR}/
  │    - skeleton_segments.nii.gz                       │
  │    - p01_multilabel.nii.gz                          │
  │    - seed_points_vis.nii.gz                         │
  │  Note segment IDs to skip, update SKIP_SEGMENTS     │
  │  Then set RUN_PART_A=False, RUN_PART_B=True         │
  └─────────────────────────────────────────────────────┘
""")

# Part B
if RUN_PART_B:
    print("\n[Step 3 Part B] Batch all phases")
    run_part_b(
        nifti_dir     = NIFTI_DIR,
        mask_path     = MASK_PATH,
        output_dir    = PLANB_OUTPUT_DIR,
        n_phases      = N_PHASES,
        skip_segments = SKIP_SEGMENTS,
    )

# Build graph
if RUN_BUILD_GRAPH:
    print("\n[Step 3] Build vessel graph + 3D HTML")
    build_graph(
        planb_output_dir   = PLANB_OUTPUT_DIR,
        out_dir            = STEP1_DIR,
        n_phases           = N_PHASES,
        skip_segments      = SKIP_SEGMENTS,
        bif_cluster_radius = BIF_CLUSTER_RADIUS,
        ep_cluster_radius  = EP_CLUSTER_RADIUS,
        mask_downsample    = MASK_DOWNSAMPLE,
    )
    print(f"  3D visualization: {STEP1_DIR}/vessel_graph_3d.html")

# Area + PI
if RUN_AREA_PI:
    print("\n[Step 3] Cross-section area + pulsatility index")
    compute_area_pi(
        step1_dir          = STEP1_DIR,
        planb_output_dir   = PLANB_OUTPUT_DIR,
        out_dir            = STEP2_DIR,
        n_phases           = N_PHASES,
        sample_spacing_mm  = SAMPLE_SPACING_MM,
        min_samples        = MIN_SAMPLES,
        max_samples        = MAX_SAMPLES,
        min_seg_length_mm  = MIN_SEG_LENGTH_MM,
        s_skip_mm          = S_SKIP_MM,
        slab_half_mm       = SLAB_HALF_MM,
        r_max_mm           = R_MAX_MM,
        recenter_radius_mm = RECENTER_RADIUS_MM,
        recenter_slab_mm   = RECENTER_SLAB_MM,
        smooth_window      = SMOOTH_WINDOW,
        path_smooth_mm     = PATH_SMOOTH_MM,
        xsec_n_slices      = XSEC_N_SLICES,
        xsec_radius_mm     = XSEC_RADIUS_MM,
        xsec_resolution    = XSEC_RESOLUTION,
        xsec_phase         = XSEC_PHASE,
    )

# Export
if RUN_EXPORT:
    if HAS_CSF:
        print("\n[Step 3] Export vessel PI + paravascular ADC → combined .mat")
        export_with_csf(
            step1_dir       = STEP1_DIR,
            step2_dir       = STEP2_DIR,
            vessel_dir      = PLANB_OUTPUT_DIR,
            adc_folder      = ADC_FOLDER,
            crop_dir        = CROP_DIR,
            out_mat         = OUT_MAT_COMBINED,
            n_phases        = N_PHASES,
            n_workers       = N_WORKERS_ADC,
            perivas_dist_mm = PERIVAS_DIST_MM,
            grid_res_mm     = GRID_RES_MM,
            grid_half_mm    = GRID_HALF_MM,
        )
    else:
        print("\n[Step 3] Export vessel PI only → .mat")
        export_vessel_only(
            step1_dir = STEP1_DIR,
            step2_dir = STEP2_DIR,
            out_mat   = OUT_MAT_VESSEL,
        )

print("\n" + "=" * 60)
print("Step 3 complete.")
print("=" * 60)