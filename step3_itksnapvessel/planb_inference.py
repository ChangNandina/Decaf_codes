#!/usr/bin/env python3
"""
planb_inference.py
Part A: prepare skeleton, seeds, Voronoi → run phase 1 inference
Part B: batch all phases using Part A outputs
"""

import requests, json, gzip, base64, os, time
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure

SERVER = "http://localhost:8912"


# ── skeleton / segment helpers ──────────────────────────────────────────────

def _edt_one_segment(args):
    sid, coords, shape, bbox_pad = args
    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < shape[0]) &
        (coords[:, 1] >= 0) & (coords[:, 1] < shape[1]) &
        (coords[:, 2] >= 0) & (coords[:, 2] < shape[2])
    )
    c = coords[valid]
    if len(c) == 0:
        return sid, np.full(shape, np.inf, dtype=np.float32)
    lo = np.maximum(c.min(axis=0) - bbox_pad, 0)
    hi = np.minimum(c.max(axis=0) + bbox_pad + 1, np.array(shape))
    marker = np.zeros(tuple(hi - lo), dtype=bool)
    c_local = c - lo
    marker[c_local[:, 0], c_local[:, 1], c_local[:, 2]] = True
    d_sub = distance_transform_edt(~marker).astype(np.float32)
    d_full = np.full(shape, np.inf, dtype=np.float32)
    d_full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = d_sub
    return sid, d_full


def compute_dist_and_voronoi(shape, segments, margin=5, n_workers=16, bbox_pad=80):
    sids = sorted(segments.keys())
    print(f"    {len(sids)} segments, {n_workers} workers")
    t0 = time.time()
    work_items = [(sid, segments[sid], shape, bbox_pad) for sid in sids]
    dist_maps = {}
    with Pool(processes=n_workers) as pool:
        for i, (sid, d_full) in enumerate(pool.imap_unordered(_edt_one_segment, work_items)):
            dist_maps[sid] = d_full
            if (i + 1) % 10 == 0 or (i + 1) == len(sids):
                print(f"      EDT: {i+1}/{len(sids)}")
    any_skel = np.zeros(shape, dtype=bool)
    for sid, coords in segments.items():
        c = coords[(coords[:, 0] < shape[0]) & (coords[:, 1] < shape[1]) & (coords[:, 2] < shape[2])]
        if len(c): any_skel[c[:, 0], c[:, 1], c[:, 2]] = True
    nearest_dist = distance_transform_edt(~any_skel).astype(np.float32)
    threshold = nearest_dist + margin
    soft_regions = {sid: dist_maps[sid] < threshold for sid in sids}
    print(f"    Voronoi done in {time.time()-t0:.1f}s")
    return dist_maps, soft_regions


def farthest_point_sampling(points, n_samples):
    n = len(points)
    if n_samples >= n: return list(range(n))
    center = points.mean(axis=0)
    first  = np.argmin(np.sum((points - center)**2, axis=1))
    selected = [first]
    min_dists = np.full(n, np.inf)
    for _ in range(n_samples - 1):
        last = points[selected[-1]]
        dists = np.sum((points - last)**2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        selected.append(np.argmax(min_dists))
    return selected


def find_branch_points(skeleton):
    struct = ndimage.generate_binary_structure(3, 3)
    nc = ndimage.convolve((skeleton > 0).astype(np.int32), struct.astype(np.int32), mode='constant') - 1
    return (skeleton > 0) & (nc >= 3)


def prune_skeleton(skeleton, min_spur_length=10):
    skel = skeleton.copy().astype(np.uint8)
    struct = ndimage.generate_binary_structure(3, 3)
    changed = True
    while changed:
        changed = False
        nc = ndimage.convolve((skel > 0).astype(np.int32), struct.astype(np.int32), mode='constant') - 1
        bp_mask  = (skel > 0) & (nc >= 3)
        tip_mask = (skel > 0) & (nc == 1)
        skel_no_bp = skel.copy(); skel_no_bp[bp_mask] = 0
        labeled, n_labels = ndimage.label(skel_no_bp, structure=struct)
        for lab in range(1, n_labels + 1):
            coords = np.argwhere(labeled == lab)
            if len(coords) >= min_spur_length: continue
            if any(tip_mask[c[0], c[1], c[2]] for c in coords):
                skel[coords[:, 0], coords[:, 1], coords[:, 2]] = 0
                changed = True
    return skel


def extract_segments(mask_data, min_length=1, bp_dilation=0, min_spur_length=3):
    binary   = (mask_data > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8)
    skeleton = prune_skeleton(skeleton, min_spur_length=min_spur_length)
    bp       = find_branch_points(skeleton)
    struct   = ndimage.generate_binary_structure(3, 3)
    bp_dil   = ndimage.binary_dilation(bp, struct, iterations=bp_dilation) if bp_dilation > 0 else bp
    skel_no_bp = skeleton.copy(); skel_no_bp[bp_dil] = 0
    labeled, n_labels = ndimage.label(skel_no_bp, structure=struct)
    segments = {}
    for i in range(1, n_labels + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) >= min_length:
            segments[len(segments) + 1] = coords
    print(f"  Segments (>={min_length} vox): {len(segments)}")
    return segments, skeleton


def allocate_seed_points(segments, total_points, min_per_seg=1):
    lengths = {sid: len(c) for sid, c in segments.items()}
    total_len = sum(lengths.values())
    allocation = {sid: max(min_per_seg, round(total_points * lengths[sid] / total_len))
                  for sid in segments}
    while sum(allocation.values()) > total_points:
        longest = max(allocation, key=allocation.get)
        if allocation[longest] > min_per_seg: allocation[longest] -= 1
        else: break
    while sum(allocation.values()) < total_points:
        allocation[max(lengths, key=lengths.get)] += 1
        if sum(allocation.values()) >= total_points: break
    return allocation


# ── inference helpers ─────────────────────────────────────────────────────────

def _decode_response(response_json, shape):
    result_raw = gzip.decompress(base64.b64decode(response_json["result"]))
    return np.frombuffer(result_raw, dtype=np.int8).reshape(shape)


def _compute_dice(a, b):
    a, b = (a > 0).astype(bool), (b > 0).astype(bool)
    inter = (a & b).sum()
    total = a.sum() + b.sum()
    return 2.0 * inter / total if total > 0 else 0.0


def _run_inference_phase(phase, nifti_dir, segment_seeds, soft_regions,
                          dist_maps, mask_shape, output_dir, mask_gt=None):
    nifti_path = os.path.join(nifti_dir, f"phase{phase}.nii.gz")
    img_nib    = nib.load(nifti_path)
    img_data   = img_nib.get_fdata().astype(np.float32)
    affine     = img_nib.affine
    voxel_vol  = np.abs(np.linalg.det(affine[:3, :3]))

    img_gz     = gzip.compress(img_data.tobytes(), compresslevel=4)
    dimensions = [int(d) for d in img_data.shape[::-1]]

    r = requests.get(f"{SERVER}/start_session")
    session_id = r.json()["session_id"]
    metadata   = json.dumps({"dimensions": dimensions})
    r = requests.post(f"{SERVER}/upload_raw/{session_id}",
                      files={"file": ("image.raw.gz", img_gz, "application/octet-stream")},
                      data={"metadata": metadata})
    del img_gz
    if r.status_code != 200:
        print(f"  Upload FAILED: {r.text[:200]}")
        requests.get(f"{SERVER}/end_session/{session_id}")
        return None, None

    clipped_masks = {}
    for sid, seeds in segment_seeds.items():
        requests.get(f"{SERVER}/reset_interactions/{session_id}")
        last_response = None
        for pt in seeds:
            r = requests.get(f"{SERVER}/process_point_interaction/{session_id}",
                             params={"x": int(pt[2]), "y": int(pt[1]),
                                     "z": int(pt[0]), "foreground": True})
            if r.status_code == 200: last_response = r.json()
        if last_response and last_response.get("status") == "success":
            raw_mask = _decode_response(last_response, mask_shape)
            clipped_masks[sid] = (raw_mask > 0) & soft_regions[sid]
        else:
            clipped_masks[sid] = np.zeros(mask_shape, dtype=bool)
    requests.get(f"{SERVER}/end_session/{session_id}")

    # Merge with distance tiebreak
    multilabel     = np.zeros(mask_shape, dtype=np.uint8)
    coverage_count = np.zeros(mask_shape, dtype=np.int32)
    for sid, mask in clipped_masks.items(): coverage_count[mask] += 1
    for sid, mask in clipped_masks.items(): multilabel[mask & (coverage_count == 1)] = sid
    overlap_zone = coverage_count >= 2
    if overlap_zone.sum() > 0:
        best_dist = np.full(mask_shape, np.inf, dtype=np.float32)
        for sid, mask in clipped_masks.items():
            contest = mask & overlap_zone
            closer  = dist_maps[sid] < best_dist
            update  = contest & closer
            multilabel[update] = sid
            best_dist[update]  = dist_maps[sid][update]

    binary_combined = (multilabel > 0)
    voxel_vol_total = binary_combined.sum() * voxel_vol

    nib.save(nib.Nifti1Image(binary_combined.astype(np.uint8), affine),
             os.path.join(output_dir, f"p{phase:02d}_binary.nii.gz"))
    nib.save(nib.Nifti1Image(multilabel, affine),
             os.path.join(output_dir, f"p{phase:02d}_multilabel.nii.gz"))

    if mask_gt is not None:
        dice = _compute_dice(binary_combined, mask_gt)
        print(f"  ★ Phase {phase} Dice: {dice:.4f}")

    phase_volumes = {sid: int((multilabel == sid).sum()) * voxel_vol
                     for sid in segment_seeds}
    print(f"  Phase {phase}: {binary_combined.sum()} voxels, {voxel_vol_total:.1f} mm³")
    return phase_volumes, {'final': binary_combined.sum()}


# ── public API ────────────────────────────────────────────────────────────────

def run_part_a(nifti_dir, mask_path, output_dir,
               total_points=100, min_branch_length=1, bp_dilation=0,
               min_spur_length=3, soft_margin=5, n_workers=16, bbox_pad=80,
               skip_segments=None):
    """Prepare skeleton/seeds/Voronoi and run phase 1 inference.
    Returns segment_seeds for review.
    """
    os.makedirs(output_dir, exist_ok=True)
    if skip_segments is None: skip_segments = []

    mask_nib   = nib.load(mask_path)
    mask_data  = mask_nib.get_fdata()
    mask_shape = mask_data.shape
    print(f"  Mask shape: {mask_shape}, fg voxels: {(mask_data > 0).sum()}")

    segments, skeleton = extract_segments(mask_data, min_length=min_branch_length,
                                          bp_dilation=bp_dilation,
                                          min_spur_length=min_spur_length)
    if not segments:
        raise RuntimeError("No segments found!")

    # Save segments
    with open(os.path.join(output_dir, "segments.json"), 'w') as f:
        json.dump({str(sid): coords.tolist() for sid, coords in segments.items()}, f)

    # Seeds (load existing or recompute)
    seeds_file = os.path.join(output_dir, "segment_seeds.json")
    if os.path.exists(seeds_file):
        print(f"  Loading existing seeds: {seeds_file}")
        with open(seeds_file) as f:
            sd = json.load(f)
        segment_seeds = {int(k): np.array(v) for k, v in sd['seeds'].items()}
    else:
        allocation = allocate_seed_points(segments, total_points)
        segment_seeds = {}
        for sid, coords in segments.items():
            n_pts = allocation[sid]
            idx   = farthest_point_sampling(coords.astype(np.float64), n_pts)
            segment_seeds[sid] = coords[idx]
        with open(seeds_file, 'w') as f:
            json.dump({'total_points': total_points,
                       'note': 'coords are [z,y,x]. Delete key to skip.',
                       'seeds': {str(sid): pts.tolist()
                                 for sid, pts in segment_seeds.items()}}, f, indent=2)
        print(f"  Seeds saved: {seeds_file}")

    for sid in skip_segments:
        if sid in segment_seeds:
            del segment_seeds[sid]
            print(f"  Skipped segment {sid}")
    print(f"  Active segments: {len(segment_seeds)}")

    # Voronoi
    dist_maps, soft_regions = compute_dist_and_voronoi(
        mask_shape, segments, margin=soft_margin, n_workers=n_workers, bbox_pad=bbox_pad)
    np.savez_compressed(os.path.join(output_dir, "dist_maps.npz"),
                        **{str(sid): dm for sid, dm in dist_maps.items()})
    np.savez_compressed(os.path.join(output_dir, "soft_regions.npz"),
                        **{str(sid): sr for sid, sr in soft_regions.items()})

    # Visualizations
    _save_visualizations(output_dir, mask_shape, mask_nib, segments,
                         segment_seeds, soft_regions, dist_maps, skeleton)

    # Phase 1 inference
    print("\n  Running Phase 1 inference...")
    _run_inference_phase(1, nifti_dir, segment_seeds, soft_regions,
                         dist_maps, mask_shape, output_dir, mask_gt=mask_data)
    print(f"\n  Part A complete. Review in ITK-SNAP:")
    print(f"    {os.path.join(output_dir, 'skeleton_segments.nii.gz')}")
    print(f"    {os.path.join(output_dir, 'p01_multilabel.nii.gz')}")
    print(f"    {os.path.join(output_dir, 'seed_points_vis.nii.gz')}")
    print(f"  Edit SKIP_SEGMENTS in run_step3.py, then set RUN_PART_B=True")
    return segment_seeds


def run_part_b(nifti_dir, mask_path, output_dir, n_phases=25, skip_segments=None):
    """Batch all phases using Part A outputs."""
    if skip_segments is None: skip_segments = []

    mask_nib   = nib.load(mask_path)
    mask_data  = mask_nib.get_fdata()
    mask_shape = mask_data.shape

    with open(os.path.join(output_dir, "segments.json")) as f:
        seg_data = json.load(f)
    segments = {int(k): np.array(v) for k, v in seg_data.items()}

    with open(os.path.join(output_dir, "segment_seeds.json")) as f:
        sd = json.load(f)
    segment_seeds = {int(k): np.array(v) for k, v in sd['seeds'].items()}
    for sid in skip_segments:
        segment_seeds.pop(sid, None)
    print(f"  Active segments: {len(segment_seeds)}")

    dm_npz = np.load(os.path.join(output_dir, "dist_maps.npz"))
    sr_npz = np.load(os.path.join(output_dir, "soft_regions.npz"))
    dist_maps    = {int(k): dm_npz[k] for k in dm_npz.files}
    soft_regions = {int(k): sr_npz[k] for k in sr_npz.files}

    all_volumes = {}
    for phase in range(1, n_phases + 1):
        if not os.path.exists(os.path.join(nifti_dir, f"phase{phase}.nii.gz")):
            print(f"  Phase {phase}: NOT FOUND, skipping")
            continue
        print(f"\n{'─'*50}\nPhase {phase}/{n_phases}")
        vols, _ = _run_inference_phase(phase, nifti_dir, segment_seeds,
                                        soft_regions, dist_maps, mask_shape, output_dir,
                                        mask_gt=mask_data if phase == 1 else None)
        if vols: all_volumes[phase] = vols

    # Save volumes CSV
    seg_ids  = sorted(segment_seeds.keys())
    csv_path = os.path.join(output_dir, "volumes.csv")
    with open(csv_path, 'w') as f:
        f.write("Phase," + ",".join([f"Seg{sid}" for sid in seg_ids]) + ",Total\n")
        for ph in sorted(all_volumes.keys()):
            vals  = [all_volumes[ph].get(sid, 0) for sid in seg_ids]
            total = sum(vals)
            f.write(f"{ph}," + ",".join(f"{v:.2f}" for v in vals) + f",{total:.2f}\n")
    print(f"\n  Volumes CSV: {csv_path}")
    print("  Part B complete.")


def _save_visualizations(output_dir, mask_shape, mask_nib, segments,
                          segment_seeds, soft_regions, dist_maps, skeleton):
    affine = mask_nib.affine
    struct = generate_binary_structure(3, 3)

    soft_vis = np.zeros(mask_shape, dtype=np.uint8)
    for sid in sorted(segments.keys()):
        already = soft_vis > 0
        soft_vis[soft_regions[sid] & ~already] = sid
        soft_vis[soft_regions[sid] & already]  = 255
    nib.save(nib.Nifti1Image(soft_vis, affine),
             os.path.join(output_dir, "soft_voronoi_regions.nii.gz"))

    point_vis = np.zeros(mask_shape, dtype=np.uint8)
    for sid, seeds in segment_seeds.items():
        for pt in seeds:
            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nz, ny, nx = int(pt[0])+dz, int(pt[1])+dy, int(pt[2])+dx
                        if all(0 <= v < s for v, s in zip([nz,ny,nx], mask_shape)):
                            point_vis[nz, ny, nx] = sid
    nib.save(nib.Nifti1Image(point_vis, affine),
             os.path.join(output_dir, "seed_points_vis.nii.gz"))

    skel_colored = np.zeros(mask_shape, dtype=np.uint8)
    for sid, coords in segments.items():
        skel_colored[coords[:, 0], coords[:, 1], coords[:, 2]] = sid
    nib.save(nib.Nifti1Image(skel_colored, affine),
             os.path.join(output_dir, "skeleton_segments.nii.gz"))

    # Fat skeleton
    skel_fat = np.zeros(mask_shape, dtype=np.uint8)
    for sid, coords in segments.items():
        lo = np.maximum(coords.min(axis=0) - 4, 0)
        hi = np.minimum(coords.max(axis=0) + 5, np.array(mask_shape))
        sub = np.zeros(tuple(hi - lo), dtype=bool)
        cl  = coords - lo
        sub[cl[:, 0], cl[:, 1], cl[:, 2]] = True
        dilated = binary_dilation(sub, struct, iterations=3)
        skel_fat[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][dilated] = sid
    nib.save(nib.Nifti1Image(skel_fat, affine),
             os.path.join(output_dir, "skeleton_segments_fat.nii.gz"))