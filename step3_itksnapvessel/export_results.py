#!/usr/bin/env python3
"""
export_results.py
Export vessel PI (and optionally paravascular ADC) to .mat for MATLAB.
HAS_CSF=True  → combined vessel+CSF mat  (uses adc_corrected_csf files)
HAS_CSF=False → vessel-only mat
"""

import os, pickle
import numpy as np
import nibabel as nib
import scipy.io as sio
import hdf5storage
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import map_coordinates, distance_transform_edt
from collections import defaultdict
from multiprocessing import get_context


# ── shared helpers ────────────────────────────────────────────────────────────

def _path_to_mm(path, affine):
    vox = np.array(path, dtype=float)
    return (affine @ np.hstack([vox, np.ones((len(vox),1))]).T).T[:,:3]

def _arc_cumlen(path_mm):
    segs = np.linalg.norm(np.diff(path_mm, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segs)])

def _interp_pos(path_mm, cumlen, s):
    s   = float(np.clip(s, cumlen[0], cumlen[-1]))
    idx = int(np.clip(np.searchsorted(cumlen, s, side='right')-1, 0, len(path_mm)-2))
    seg_len = cumlen[idx+1]-cumlen[idx]
    if seg_len < 1e-9: return path_mm[idx].copy()
    return path_mm[idx] + (s-cumlen[idx])/seg_len*(path_mm[idx+1]-path_mm[idx])

def _extract_ref_segments(ref_graph):
    nids = {tuple(int(x) for x in k): v for k, v in ref_graph['node_ids'].items()}
    segments = {}
    for key, edge in ref_graph['edges'].items():
        path = edge['path']
        if len(path) < 2: continue
        a=tuple(int(x) for x in path[0]); b=tuple(int(x) for x in path[-1])
        a_id=nids.get(a,str(a)); b_id=nids.get(b,str(b))
        a_bif=a_id.startswith('BIF'); b_bif=b_id.startswith('BIF')
        if a_bif and b_bif:
            seg_type='bif-bif'
            if a_id<=b_id: na,nb,po=a_id,b_id,list(path)
            else:          na,nb,po=b_id,a_id,list(reversed(path))
            seg_id=f"{na}—{nb}"
        elif a_bif:  seg_type='bif-ep';  na,nb,po=a_id,b_id,list(path);          seg_id=f"{na}→{nb}"
        elif b_bif:  seg_type='bif-ep';  na,nb,po=b_id,a_id,list(reversed(path)); seg_id=f"{na}→{nb}"
        else:
            seg_type='ep-ep'
            if a_id<=b_id: na,nb,po=a_id,b_id,list(path)
            else:          na,nb,po=b_id,a_id,list(reversed(path))
            seg_id=f"{na}—{nb}"
        if seg_id not in segments:
            segments[seg_id]={'node_a':na,'node_b':nb,'seg_type':seg_type,
                               'length_mm':edge['length_mm'],'ref_path':po}
    return segments


# ── CSF ADC helpers ───────────────────────────────────────────────────────────

def _load_adc_cropped(adc_folder, phase, crop_origin, crop_size, adc_permute=(1,2,0)):
    import h5py
    fname = os.path.join(adc_folder, f'adc_corrected_csf_phase{phase:02d}_single.mat')
    with h5py.File(fname, 'r') as f:
        adc_raw = f['adcCorrectedCSF'][()].astype(np.float32)
    adc = np.transpose(adc_raw, adc_permute)
    o,s = crop_origin, crop_size
    return adc[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy()

def _get_plane_basis(tang):
    t = tang/np.linalg.norm(tang)
    ref = np.array([1.,0.,0.]) if abs(np.dot(t,[1.,0.,0.])) <= 0.9 else np.array([0.,1.,0.])
    e1 = np.cross(t,ref); e1/=np.linalg.norm(e1)
    e2 = np.cross(t,e1);  e2/=np.linalg.norm(e2)
    return e1, e2

def _sample_plane(vol, center_mm, e1, e2, inv_affine, half_mm, resolution):
    n   = int(2*half_mm/resolution)+1
    lin = np.linspace(-half_mm, half_mm, n)
    g1,g2 = np.meshgrid(lin, lin, indexing='xy')
    pts_mm   = center_mm[None,None,:] + g1[...,None]*e1[None,None,:] + g2[...,None]*e2[None,None,:]
    pts_flat = pts_mm.reshape(-1,3)
    pts_vox  = (inv_affine @ np.hstack([pts_flat, np.ones((len(pts_flat),1))]).T).T[:,:3].reshape(n,n,3)
    patch = map_coordinates(vol, [pts_vox[...,i].ravel() for i in range(3)],
                            order=1, mode='constant', cval=0.0).reshape(n,n)
    return patch, lin

def _extract_csf_adc(pos_mm, tang_unit, adc_vol, adc_inv_affine,
                      vessel_vol, vessel_inv_affine,
                      perivas_dist_mm=3.0, grid_res_mm=0.2, grid_half_mm=6.0):
    e1,e2 = _get_plane_basis(tang_unit)
    vessel_patch, lin = _sample_plane(vessel_vol, pos_mm, e1, e2, vessel_inv_affine,
                                       grid_half_mm, grid_res_mm)
    vessel_bin = vessel_patch > 0.5
    if not np.any(vessel_bin): return np.nan, 0
    dist_mm = distance_transform_edt(vessel_bin==0) * grid_res_mm
    shell   = (vessel_bin==0) & (dist_mm <= perivas_dist_mm)
    if not np.any(shell): return np.nan, 0
    idx      = np.argwhere(shell)
    d1,d2    = lin[idx[:,0]], lin[idx[:,1]]
    shell_mm = pos_mm[None,:] + d1[:,None]*e1[None,:] + d2[:,None]*e2[None,:]
    ones     = np.ones((len(shell_mm),1))
    shell_vox = (adc_inv_affine @ np.hstack([shell_mm,ones]).T).T[:,:3]
    shape     = adc_vol.shape
    inb       = ((shell_vox[:,0]>=0)&(shell_vox[:,0]<shape[0])&
                  (shell_vox[:,1]>=0)&(shell_vox[:,1]<shape[1])&
                  (shell_vox[:,2]>=0)&(shell_vox[:,2]<shape[2]))
    if not np.any(inb): return np.nan, 0
    adc_vals = map_coordinates(adc_vol, [shell_vox[inb,i] for i in range(3)],
                                order=1, mode='constant', cval=0.0)
    valid = adc_vals > 0
    return (float(np.nanmean(adc_vals[valid])), int(valid.sum())) if valid.any() else (np.nan, 0)

def _process_phase_adc(args):
    (ph, all_pos_mm, all_tang, seg_label_map, seg_to_indices, N_total,
     crop_origin, crop_size, adc_inv_affine, vessel_dir, adc_folder,
     perivas_dist_mm, grid_res_mm, grid_half_mm, all_seg_ids) = args
    adc_vol       = _load_adc_cropped(adc_folder, ph, crop_origin, crop_size)
    vessel_nib    = nib.load(os.path.join(vessel_dir, f'p{ph:02d}_multilabel.nii.gz'))
    vessel_ml     = vessel_nib.get_fdata().astype(np.uint8)
    vessel_inv    = np.linalg.inv(vessel_nib.affine)
    adc_col  = np.full(N_total, np.nan, dtype=np.float32)
    nvox_col = np.zeros(N_total, dtype=np.int32)
    for seg_id, idx_list in seg_to_indices.items():
        seg_label = seg_label_map.get(seg_id)
        if seg_label is None: continue
        vessel_seg_f = (vessel_ml == seg_label).astype(np.float32)
        for i in idx_list:
            adc_mean, n_vox = _extract_csf_adc(
                all_pos_mm[i], all_tang[i], adc_vol, adc_inv_affine,
                vessel_seg_f, vessel_inv, perivas_dist_mm, grid_res_mm, grid_half_mm)
            adc_col[i]=adc_mean; nvox_col[i]=n_vox
    return ph, adc_col, nvox_col


# ── public API ────────────────────────────────────────────────────────────────

def export_vessel_only(step1_dir, step2_dir, out_mat):
    """Export vessel PI only → .mat"""
    with open(os.path.join(step1_dir, "reference_graph.pkl"), "rb") as f:
        ref = pickle.load(f)
    with open(os.path.join(step2_dir, "pi_results.pkl"), "rb") as f:
        step2 = pickle.load(f)
    pi_all = step2['pi']
    ref_affine   = ref['affine']
    ref_segments = _extract_ref_segments(ref)

    all_pts, all_areas, all_pi_vals, all_seg = [], [], [], []
    for seg_id, r in sorted(pi_all.items()):
        if r is None or seg_id not in ref_segments: continue
        pmm    = _path_to_mm(ref_segments[seg_id]['ref_path'], ref_affine)
        cumlen = _arc_cumlen(pmm)
        for si, s in enumerate(r['s_vals']):
            if s > cumlen[-1]+1e-6: continue
            areas_25 = np.array([r['area_by_ph'][ph][si] if si < len(r['area_by_ph'][ph]) else 0.0
                                  for ph in range(1,26)])
            all_pts.append(_interp_pos(pmm, cumlen, s))
            all_areas.append(areas_25)
            all_pi_vals.append(r['pi_per_s'][si] if si < len(r['pi_per_s']) else np.nan)
            all_seg.append(seg_id)

    cl_mm    = np.array(all_pts)
    area_all = np.array(all_areas)
    pi_val   = np.array(all_pi_vals)
    sio.savemat(out_mat, {'cl_mm': cl_mm, 'area_all': area_all,
                           'pi_val': pi_val, 'seg_ids': np.array(all_seg, dtype=object),
                           'affine': ref_affine, 'n_phases': 25})
    print(f"  Vessel-only mat saved: {out_mat}")
    print(f"  Points: {cl_mm.shape[0]}, PI range: [{np.nanmin(pi_val):.4f}, {np.nanmax(pi_val):.4f}]")


def export_with_csf(step1_dir, step2_dir, vessel_dir, adc_folder, crop_dir,
                     out_mat, n_phases=25, n_workers=6,
                     perivas_dist_mm=3.0, grid_res_mm=0.2, grid_half_mm=6.0):
    """Export combined vessel PI + paravascular ADC → .mat"""
    import json
    with open(os.path.join(step1_dir, "reference_graph.pkl"), "rb") as f:
        ref = pickle.load(f)
    with open(os.path.join(step2_dir, "pi_results.pkl"), "rb") as f:
        step2 = pickle.load(f)
    pi_all = step2['pi']; seg_label_map = step2['seg_label_map']
    ref_affine = ref['affine']

    # Crop info
    with open(os.path.join(crop_dir, 'crop_info.json')) as f:
        crop_info = json.load(f)
    import glob
    nii_files = glob.glob(os.path.join(crop_dir, '*.nii.gz'))
    crop_affine = nib.load(nii_files[0]).affine.copy()
    adc_inv_affine = np.linalg.inv(crop_affine)
    crop_origin = crop_info['crop_origin']
    crop_size   = crop_info['crop_size']

    # Collect sample points
    all_seg_ids, all_pos_mm, all_tang = [], [], []
    all_area_mat, all_pi_val = [], []
    for seg_id, r in sorted(pi_all.items()):
        if r is None: continue
        for si in range(r['n_samples']):
            all_seg_ids.append(seg_id)
            all_pos_mm.append(r['recen_pts_mm'][si])
            all_tang.append(r['tangents'][si])
            all_area_mat.append(r['area_mat'][:,si])
            all_pi_val.append(r['pi_per_s'][si])
    N_total    = len(all_pos_mm)
    all_pos_mm = np.asarray(all_pos_mm, dtype=np.float64)
    all_tang   = np.asarray(all_tang,   dtype=np.float64)
    area_out   = np.array(all_area_mat, dtype=np.float32)
    pi_out     = np.array(all_pi_val,   dtype=np.float32)
    cl_mm_out  = all_pos_mm.astype(np.float32)
    seg_to_indices = defaultdict(list)
    for i, sid in enumerate(all_seg_ids): seg_to_indices[sid].append(i)

    # ADC per phase (parallel)
    print(f"  Computing paravascular ADC ({n_workers} workers, {N_total} points)...")
    job_args = [(ph, all_pos_mm, all_tang, seg_label_map, dict(seg_to_indices), N_total,
                  crop_origin, crop_size, adc_inv_affine, vessel_dir, adc_folder,
                  perivas_dist_mm, grid_res_mm, grid_half_mm, all_seg_ids)
                for ph in range(1, n_phases+1)]
    adc_out  = np.full((N_total, n_phases), np.nan, dtype=np.float32)
    nvox_out = np.zeros((N_total, n_phases), dtype=np.int32)
    ctx = get_context('fork')
    with ctx.Pool(processes=n_workers) as pool:
        for ph, adc_col, nvox_col in pool.imap_unordered(_process_phase_adc, job_args):
            adc_out[:,ph-1]=adc_col; nvox_out[:,ph-1]=nvox_col
            print(f"  Phase {ph}/{n_phases} done")

    sio.savemat(out_mat, {'cl_mm': cl_mm_out, 'area_all': area_out, 'pi_val': pi_out,
                           'adc_all': adc_out, 'adc_nvox': nvox_out,
                           'seg_ids': np.array(all_seg_ids, dtype=object),
                           'affine': crop_affine, 'n_phases': n_phases})
    print(f"  Combined vessel+CSF mat saved: {out_mat}")
    valid = adc_out[~np.isnan(adc_out)]
    if len(valid):
        print(f"  ADC range: [{valid.min():.6f}, {valid.max():.6f}]")