#!/usr/bin/env python3
"""area_pi.py — Cross-section area and pulsatility index per vessel segment."""

import os, pickle, time
import numpy as np
import nibabel as nib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.ndimage import map_coordinates, distance_transform_edt, gaussian_filter1d


# ── path / geometry helpers ───────────────────────────────────────────────────

def _path_to_mm(path, affine):
    vox  = np.array(path, dtype=float)
    ones = np.ones((len(vox), 1))
    return (affine @ np.hstack([vox, ones]).T).T[:, :3]


def _arc_cumlen(path_mm):
    segs = np.linalg.norm(np.diff(path_mm, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segs)])


def _interp_at_arc(path_mm, cumlen, s):
    s   = float(np.clip(s, cumlen[0], cumlen[-1]))
    idx = int(np.clip(np.searchsorted(cumlen, s, side='right') - 1, 0, len(path_mm)-2))
    seg_len = cumlen[idx+1] - cumlen[idx]
    if seg_len < 1e-9:
        return path_mm[idx].copy(), np.array([1.,0.,0.])
    t    = (s - cumlen[idx]) / seg_len
    d    = path_mm[idx+1] - path_mm[idx]
    return path_mm[idx] + t*d, d/np.linalg.norm(d)


def _smooth_path(path_mm, sigma_mm=1.0):
    cumlen = _arc_cumlen(path_mm)
    if cumlen[-1] < 1e-6 or len(path_mm) < 5: return path_mm.copy()
    avg_step = cumlen[-1] / (len(path_mm)-1)
    sigma_s  = max(sigma_mm / max(avg_step, 1e-9), 0.5)
    smoothed = np.stack([gaussian_filter1d(path_mm[:,ax], sigma_s) for ax in range(3)], axis=1)
    smoothed[0] = path_mm[0]; smoothed[-1] = path_mm[-1]
    return smoothed


def _smooth_1d(arr, window):
    if window <= 1: return arr.copy()
    if window % 2 == 0: window += 1
    n, half, out = len(arr), window//2, np.full(len(arr), np.nan)
    for i in range(n):
        chunk = arr[max(0,i-half):min(n,i+half+1)]
        valid = chunk[~np.isnan(chunk)]
        if len(valid): out[i] = np.mean(valid)
    return out


# ── label mapping ─────────────────────────────────────────────────────────────

def _find_segment_label(ref_path, multilabel_p1):
    votes = {}
    for vox in ref_path:
        z,y,x = int(round(vox[0])),int(round(vox[1])),int(round(vox[2]))
        if all(0<=c<s for c,s in zip([z,y,x],multilabel_p1.shape)):
            lab = int(multilabel_p1[z,y,x])
            if lab > 0: votes[lab] = votes.get(lab,0)+1
    return max(votes, key=votes.get) if votes else None


# ── area computation ──────────────────────────────────────────────────────────

def _recenter_and_area(pos_mm, tang_unit, multilabel, seg_label,
                        inv_affine, affine, voxel_size,
                        rc_radius_mm, rc_slab_mm, area_r_mm, area_slab_half_mm):
    max_r   = max(rc_radius_mm, area_r_mm)
    pos_vox = (inv_affine @ np.append(pos_mm, 1.0))[:3]
    r_vox   = int(np.ceil(max_r / np.min(voxel_size))) + 2
    shape   = multilabel.shape
    iz,iy,ix = int(round(pos_vox[0])),int(round(pos_vox[1])),int(round(pos_vox[2]))
    zs = slice(max(0,iz-r_vox), min(shape[0],iz+r_vox+1))
    ys = slice(max(0,iy-r_vox), min(shape[1],iy+r_vox+1))
    xs = slice(max(0,ix-r_vox), min(shape[2],ix+r_vox+1))
    local = (multilabel[zs,ys,xs] == seg_label)
    if not local.sum(): return pos_mm.copy(), 0.0
    origin = np.array([zs.start, ys.start, xs.start], dtype=float)
    gv     = np.argwhere(local).astype(float) + origin
    pts_mm = (affine @ np.hstack([gv, np.ones((len(gv),1))]).T).T[:,:3]
    offsets = pts_mm - pos_mm
    along   = offsets @ tang_unit
    recentered_pos = pos_mm.copy()
    rc_in_slab = np.abs(along) <= rc_slab_mm
    if np.any(rc_in_slab):
        perp = offsets[rc_in_slab] - along[rc_in_slab,np.newaxis]*tang_unit
        in_r = np.linalg.norm(perp, axis=1) <= rc_radius_mm
        if np.any(in_r): recentered_pos = pos_mm + perp[in_r].mean(axis=0)
    offsets2 = pts_mm - recentered_pos
    along2   = offsets2 @ tang_unit
    in_slab  = np.abs(along2) <= area_slab_half_mm
    if not np.any(in_slab): return recentered_pos, 0.0
    perp2 = offsets2[in_slab] - along2[in_slab,np.newaxis]*tang_unit
    count = int((np.sum(perp2**2, axis=1) <= area_r_mm**2).sum())
    area  = count * float(np.prod(voxel_size)) / (2.0 * area_slab_half_mm)
    return recentered_pos, area


# ── segment extraction from graph ────────────────────────────────────────────

def _extract_ref_segments(ref_graph):
    nids = {tuple(int(x) for x in k): v for k, v in ref_graph['node_ids'].items()}
    segments = {}
    for key, edge in ref_graph['edges'].items():
        path = edge['path']
        if len(path) < 2: continue
        a = tuple(int(x) for x in path[0])
        b = tuple(int(x) for x in path[-1])
        a_id = nids.get(a, str(a)); b_id = nids.get(b, str(b))
        a_bif = a_id.startswith('BIF'); b_bif = b_id.startswith('BIF')
        if a_bif and b_bif:
            seg_type = 'bif-bif'
            if a_id <= b_id: na,nb,po = a_id,b_id,list(path)
            else:            na,nb,po = b_id,a_id,list(reversed(path))
            seg_id = f"{na}—{nb}"
        elif a_bif:  seg_type='bif-ep';  na,nb,po=a_id,b_id,list(path);         seg_id=f"{na}→{nb}"
        elif b_bif:  seg_type='bif-ep';  na,nb,po=b_id,a_id,list(reversed(path));seg_id=f"{na}→{nb}"
        else:
            seg_type = 'ep-ep'
            if a_id <= b_id: na,nb,po = a_id,b_id,list(path)
            else:            na,nb,po = b_id,a_id,list(reversed(path))
            seg_id = f"{na}—{nb}"
        if seg_id not in segments:
            segments[seg_id] = {'node_a': na, 'node_b': nb, 'seg_type': seg_type,
                                 'length_mm': edge['length_mm'], 'ref_path': po}
    return segments


# ── public API ────────────────────────────────────────────────────────────────

def compute_area_pi(step1_dir, planb_output_dir, out_dir, n_phases=25,
                    sample_spacing_mm=0.5, min_samples=5, max_samples=150,
                    min_seg_length_mm=3.0, s_skip_mm=2.0,
                    slab_half_mm=0.5, r_max_mm=20.0,
                    recenter_radius_mm=2.5, recenter_slab_mm=1.0,
                    smooth_window=15, path_smooth_mm=1.0,
                    xsec_n_slices=9, xsec_radius_mm=5.0, xsec_resolution=0.2,
                    xsec_phase=1):
    """Compute cross-section areas and PI for all vessel segments across phases."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    xsec_dir = os.path.join(out_dir, "cross_sections")
    Path(xsec_dir).mkdir(parents=True, exist_ok=True)
    t_global = time.time()

    with open(os.path.join(step1_dir, "reference_graph.pkl"), "rb") as f:
        ref = pickle.load(f)
    ref_affine   = ref['affine']
    ref_segments = _extract_ref_segments(ref)

    kept    = {sid: s for sid, s in ref_segments.items() if s['length_mm'] >= min_seg_length_mm}
    skipped = {sid: s for sid, s in ref_segments.items() if s['length_mm'] <  min_seg_length_mm}
    print(f"  Segments: {len(kept)} kept, {len(skipped)} skipped (<{min_seg_length_mm}mm)")

    # Label mapping from phase 1 multilabel
    p1_ml = nib.load(os.path.join(planb_output_dir, "p01_multilabel.nii.gz")).get_fdata().astype(np.uint8)
    seg_label_map = {}
    for seg_id, seg in sorted(kept.items()):
        label = _find_segment_label(seg['ref_path'], p1_ml)
        seg_label_map[seg_id] = label
        print(f"  {seg_id:30s} → label {label}")
    del p1_ml

    # Load all phase multilabel masks
    phase_data = {}
    for ph in range(1, n_phases+1):
        fname = os.path.join(planb_output_dir, f"p{ph:02d}_multilabel.nii.gz")
        img  = nib.load(fname)
        ml   = img.get_fdata().astype(np.uint8)
        aff  = img.affine
        phase_data[ph] = {'multilabel': ml, 'affine': aff,
                           'inv_affine': np.linalg.inv(aff),
                           'voxel_size': np.abs(np.diag(aff)[:3])}
    print(f"  Loaded {n_phases} phase masks")

    all_pi = {}
    for seg_id, seg in sorted(kept.items()):
        t_seg = time.time()
        ref_path    = seg['ref_path']
        ref_path_mm = _smooth_path(_path_to_mm(ref_path, ref_affine), path_smooth_mm)
        cumlen      = _arc_cumlen(ref_path_mm)
        total_len   = cumlen[-1]
        seg_label   = seg_label_map.get(seg_id)

        s_start = min(s_skip_mm, total_len*0.1)
        s_end   = max(total_len - s_skip_mm, total_len*0.9)
        if s_end <= s_start + 1e-6: s_start, s_end = 0.0, total_len
        usable_len = s_end - s_start
        n_samples  = max(min_samples, min(max_samples, int(round(usable_len/sample_spacing_mm))))
        s_vals     = np.linspace(s_start, s_end, n_samples)
        spacing    = usable_len / max(n_samples-1, 1)

        ref_info = [_interp_at_arc(ref_path_mm, cumlen, s) for s in s_vals]
        orig_pts_mm  = np.array([pos  for pos, _ in ref_info])
        tangents_arr = np.array([tang for _, tang in ref_info])
        recen_pts    = np.zeros((n_samples, 3))

        print(f"\n  {seg_id}  [{seg['seg_type']}  {total_len:.1f}mm  label={seg_label}  n={n_samples}]")

        area_raw_by_ph = {}
        for ph in range(1, n_phases+1):
            pd = phase_data[ph]
            areas = np.zeros(n_samples)
            for si, (ref_pos, ref_tang) in enumerate(ref_info):
                if seg_label is not None:
                    rc_pos, area = _recenter_and_area(
                        ref_pos, ref_tang, pd['multilabel'], seg_label,
                        pd['inv_affine'], pd['affine'], pd['voxel_size'],
                        recenter_radius_mm, recenter_slab_mm, r_max_mm, slab_half_mm)
                else:
                    rc_pos, area = ref_pos.copy(), 0.0
                areas[si] = area
                if ph == xsec_phase: recen_pts[si] = rc_pos
            area_raw_by_ph[ph] = areas

        sw = max(3, int(round(smooth_window * n_samples / 50)))
        area_by_ph = {ph: _smooth_1d(area_raw_by_ph[ph], sw) for ph in range(1, n_phases+1)}
        area_mat   = np.vstack([area_by_ph[ph] for ph in range(1, n_phases+1)])
        mean_area  = np.nanmean(area_mat, axis=0)
        pi_per_s   = np.full(n_samples, np.nan)
        valid_s    = mean_area > 1e-6
        if np.any(valid_s):
            pi_per_s[valid_s] = ((np.nanmax(area_mat[:,valid_s], axis=0) -
                                   np.nanmin(area_mat[:,valid_s], axis=0)) / mean_area[valid_s])
        pi_mean = float(np.nanmean(pi_per_s))
        print(f"    PI mean={pi_mean:.4f}  area mean={np.nanmean(area_mat):.1f}mm²  ({time.time()-t_seg:.1f}s)")

        all_pi[seg_id] = {'seg_id': seg_id, 'seg_type': seg['seg_type'],
                           'length_mm': total_len, 'seg_label': seg_label,
                           'n_samples': n_samples, 'spacing_mm': spacing, 's_vals': s_vals,
                           'area_raw_by_ph': area_raw_by_ph, 'area_by_ph': area_by_ph,
                           'area_mat': area_mat, 'pi_per_s': pi_per_s, 'pi_mean': pi_mean,
                           'orig_pts_mm': orig_pts_mm, 'tangents': tangents_arr,
                           'recen_pts_mm': recen_pts}

    out_path = os.path.join(out_dir, "pi_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({'pi': all_pi, 'seg_label_map': seg_label_map,
                     'skipped': {sid: s['length_mm'] for sid, s in skipped.items()},
                     'config': {'sample_spacing_mm': sample_spacing_mm,
                                'min_seg_length_mm': min_seg_length_mm,
                                's_skip_mm': s_skip_mm, 'slab_half_mm': slab_half_mm,
                                'r_max_mm': r_max_mm, 'recenter_radius_mm': recenter_radius_mm}}, f)
    print(f"\n  PI results saved: {out_path}")
    print(f"  Total time: {time.time()-t_global:.1f}s")