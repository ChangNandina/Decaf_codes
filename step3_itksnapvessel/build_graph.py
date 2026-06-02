#!/usr/bin/env python3
"""
build_graph.py
Build vessel graph from Plan B segments + generate 3D HTML visualization.
"""

import os, pickle, json
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import generate_binary_structure, zoom
from skimage import measure
from pathlib import Path

STRUCT_26 = generate_binary_structure(3, 3)
KERNEL_26 = np.ones((3,3,3), dtype=np.uint8); KERNEL_26[1,1,1] = 0


# ── graph helpers ─────────────────────────────────────────────────────────────

def _classify_voxels(skel):
    skel_u8 = skel.astype(np.uint8)
    nc = ndimage.convolve(skel_u8, KERNEL_26, mode='constant', cval=0) * skel_u8
    return list(map(tuple, np.argwhere(nc == 1))), list(map(tuple, np.argwhere(nc >= 3))), nc


def _cluster_voxels(voxels, radius):
    if not voxels: return []
    arr  = np.array(voxels, dtype=float)
    used = np.zeros(len(arr), dtype=bool)
    clusters = []
    for i in range(len(arr)):
        if used[i]: continue
        dists = np.linalg.norm(arr - arr[i], axis=1)
        idx   = np.where((dists <= radius) & ~used)[0]
        used[idx] = True
        centroid = arr[idx].mean(axis=0)
        closest  = idx[np.argmin(np.linalg.norm(arr[idx] - centroid, axis=1))]
        clusters.append((tuple(arr[closest].astype(int)),
                         [tuple(arr[j].astype(int)) for j in idx]))
    return clusters


def _get_neighbors(z, y, x, shape):
    Z, Y, X = shape
    for dz in (-1,0,1):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dz==dy==dx==0: continue
                nz,ny,nx = z+dz,y+dy,x+dx
                if 0<=nz<Z and 0<=ny<Y and 0<=nx<X:
                    yield (nz,ny,nx)


def _trace_segment(start, first_step, skel_set, node_set, raw_to_rep, shape):
    path    = [start, first_step]
    visited = {start, first_step}
    prev, curr = start, first_step
    for _ in range(50000):
        if curr in raw_to_rep:
            rep = raw_to_rep[curr]
            if rep != start and rep in node_set:
                path[-1] = rep; return rep, path
        if curr in node_set and curr != start:
            return curr, path
        nbrs = [nb for nb in _get_neighbors(*curr, shape)
                if nb in skel_set and nb not in visited]
        if not nbrs: return None, path
        if len(nbrs) == 1:
            next_v = nbrs[0]
        else:
            d = np.array(curr) - np.array(prev)
            best, best_dot = nbrs[0], -999
            for nb in nbrs:
                dot = np.dot(np.array(nb)-np.array(curr), d)
                if dot > best_dot: best_dot=dot; best=nb
            next_v = best
        visited.add(next_v); prev,curr = curr,next_v; path.append(curr)
    return None, path


def _arc_length(path, vox_size):
    if len(path) < 2: return 0.0
    p = np.array(path, dtype=float) * np.array(vox_size)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _extract_graph(skel, bif_clusters, ep_clusters, vox_size):
    shape      = skel.shape
    node_set   = set(); raw_to_rep = {}
    for rep, members in bif_clusters + ep_clusters:
        node_set.add(rep)
        for m in members: raw_to_rep[m] = rep
    skel_set = set(map(tuple, np.argwhere(skel)))
    node_ids = {rep: f"BIF_{i:02d}" for i,(rep,_) in enumerate(bif_clusters)}
    node_ids.update({rep: f"EP_{i:02d}" for i,(rep,_) in enumerate(ep_clusters)})
    edges = {}
    for start in node_set:
        for first_step in _get_neighbors(*start, shape):
            if first_step not in skel_set: continue
            if first_step in raw_to_rep and raw_to_rep[first_step] == start: continue
            end, path = _trace_segment(start, first_step, skel_set, node_set, raw_to_rep, shape)
            if end is None or end == start: continue
            key = frozenset([start, end])
            new_len = _arc_length(path, vox_size)
            if key not in edges or new_len < edges[key]['length_mm']:
                edges[key] = {'path': path,
                               'node_ids': (node_ids.get(start, str(start)),
                                            node_ids.get(end, str(end))),
                               'length_mm': new_len}
    return node_ids, edges


# ── surface mesh ──────────────────────────────────────────────────────────────

def _surface_mesh(mask, affine, downsample=2):
    small = zoom(mask.astype(float), 1.0/downsample, order=1) > 0.5 if downsample > 1 else mask
    if not small.sum(): return np.zeros((0,3)), np.zeros((0,3),dtype=int)
    verts, faces, _, _ = measure.marching_cubes(small, level=0.5)
    verts_mm = (affine @ np.hstack([verts*downsample, np.ones((len(verts),1))]).T).T[:,:3]
    return verts_mm, faces


# ── HTML generation ───────────────────────────────────────────────────────────

def _generate_html(verts, faces, nodes, edge_paths, out_path):
    center = verts.mean(axis=0).tolist() if len(verts) else [0,0,0]
    mesh_data  = {'vertices': verts.tolist(), 'faces': faces.tolist(), 'center': center}
    nodes_data = [{'pos': n['pos'], 'label': n['label'], 'type': n['type']} for n in nodes]
    edges_data = [{'points': e['points'], 'label': e['label'], 'color': e['color']} for e in edge_paths]

    PAL = ['#ff6b6b','#4ecdc4','#45b7d1','#96ceb4','#ffeaa7','#dfe6e9','#fd79a8','#a29bfe',
           '#00b894','#e17055','#74b9ff','#55efc4','#fab1a0','#81ecec','#b2bec3','#6c5ce7',
           '#fdcb6e','#e84393','#00cec9','#ff7675']

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Vessel Graph 3D</title>
<style>
* {{ margin:0;padding:0;box-sizing:border-box }}
body {{ background:#0a0a0f;overflow:hidden;font-family:monospace }}
canvas {{ display:block }}
#info {{ position:absolute;top:16px;left:16px;color:#8899aa;font-size:12px;
  background:rgba(10,10,15,0.85);padding:12px 16px;border:1px solid #223;
  border-radius:6px;max-height:90vh;overflow-y:auto }}
#info h2 {{ color:#ddeeff;font-size:14px;margin-bottom:8px }}
.slider-row {{ display:flex;align-items:center;gap:8px;margin:6px 0;color:#aab }}
.slider-row input[type=range] {{ width:120px }}
.dot {{ width:10px;height:10px;border-radius:50%;display:inline-block }}
.legend-item {{ display:flex;align-items:center;gap:8px;margin:3px 0 }}
#controls {{ position:absolute;bottom:16px;left:16px;color:#667;font-size:11px;
  background:rgba(10,10,15,0.7);padding:8px 12px;border-radius:4px }}
</style></head><body>
<div id="info">
  <h2>Vessel Graph 3D</h2>
  <div class="slider-row"><label style="min-width:80px">Vessel opacity</label>
    <input type="range" id="opacitySlider" min="0" max="100" value="15">
    <span id="opacityVal">0.15</span></div>
  <div class="slider-row"><label style="min-width:80px">Edge width</label>
    <input type="range" id="widthSlider" min="1" max="10" value="3">
    <span id="widthVal">3</span></div>
  <div class="slider-row"><label style="min-width:80px">Node size</label>
    <input type="range" id="nodeSizeSlider" min="1" max="20" value="8">
    <span id="nodeSizeVal">0.8</span></div>
  <div class="slider-row"><label style="min-width:80px">Show labels</label>
    <input type="checkbox" id="labelToggle" checked></div>
  <div id="legendContainer"></div>
</div>
<div id="controls">Drag=rotate · Scroll=zoom · Right-drag=pan</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const MESH={json.dumps(mesh_data)},NODES={json.dumps(nodes_data)},EDGES={json.dumps(edges_data)};
const PAL={json.dumps(PAL)};
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0a0a0f);
const camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.1,5000);
const renderer=new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0x404060,0.6));
const d1=new THREE.DirectionalLight(0xffffff,0.8);d1.position.set(1,1,1);scene.add(d1);
const cx=MESH.center[0],cy=MESH.center[1],cz=MESH.center[2];
let vesselMesh=null;
if(MESH.vertices.length>0){{
  const g=new THREE.BufferGeometry();
  const v=new Float32Array(MESH.vertices.length*3);
  MESH.vertices.forEach((p,i)=>{{v[i*3]=p[0]-cx;v[i*3+1]=p[1]-cy;v[i*3+2]=p[2]-cz;}});
  g.setAttribute('position',new THREE.BufferAttribute(v,3));
  g.setIndex(MESH.faces.flat());g.computeVertexNormals();
  vesselMesh=new THREE.Mesh(g,new THREE.MeshPhongMaterial({{color:0x6688bb,transparent:true,opacity:0.15,side:THREE.DoubleSide,depthWrite:false}}));
  scene.add(vesselMesh);
}}
const edgeMats=[];
EDGES.forEach((e,i)=>{{
  const pts=e.points.map(p=>new THREE.Vector3(p[0]-cx,p[1]-cy,p[2]-cz));
  const mat=new THREE.LineBasicMaterial({{color:e.color||PAL[i%PAL.length],linewidth:3}});
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),mat));
  edgeMats.push(mat);
}});
const nodeGroup=new THREE.Group();const labelSprites=[];
NODES.forEach(n=>{{
  const color=n.type==='BIF'?0xff3333:0x3399ff;
  const m=new THREE.Mesh(new THREE.SphereGeometry(0.8,16,16),
    new THREE.MeshPhongMaterial({{color,emissive:color,emissiveIntensity:0.3}}));
  m.position.set(n.pos[0]-cx,n.pos[1]-cy,n.pos[2]-cz);nodeGroup.add(m);
  const c=document.createElement('canvas');c.width=256;c.height=64;
  const ctx=c.getContext('2d');ctx.font='bold 36px monospace';
  ctx.fillStyle=n.type==='BIF'?'#ff6666':'#66aaff';ctx.textAlign='center';
  ctx.fillText(n.label,128,44);
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({{map:new THREE.CanvasTexture(c),transparent:true,depthTest:false}}));
  sp.position.set(n.pos[0]-cx,n.pos[1]-cy+2,n.pos[2]-cz);sp.scale.set(8,2,1);
  nodeGroup.add(sp);labelSprites.push(sp);
}});
scene.add(nodeGroup);
let minP=new THREE.Vector3(Infinity,Infinity,Infinity),maxP=new THREE.Vector3(-Infinity,-Infinity,-Infinity);
NODES.forEach(n=>{{minP.x=Math.min(minP.x,n.pos[0]-cx);minP.y=Math.min(minP.y,n.pos[1]-cy);minP.z=Math.min(minP.z,n.pos[2]-cz);maxP.x=Math.max(maxP.x,n.pos[0]-cx);maxP.y=Math.max(maxP.y,n.pos[1]-cy);maxP.z=Math.max(maxP.z,n.pos[2]-cz);}});
const extent=maxP.clone().sub(minP).length();
let sph={{theta:0,phi:Math.PI/2,radius:extent*1.5}},pan=new THREE.Vector3(),isDrag=false,isRight=false,prev={{x:0,y:0}};
function updateCam(){{const r=sph.radius,phi=sph.phi,th=sph.theta;camera.position.set(r*Math.sin(phi)*Math.sin(th)+pan.x,r*Math.cos(phi)+pan.y,r*Math.sin(phi)*Math.cos(th)+pan.z);camera.lookAt(pan);}}
updateCam();
renderer.domElement.addEventListener('mousedown',e=>{{if(e.button===2)isRight=true;else isDrag=true;prev={{x:e.clientX,y:e.clientY}};}});
renderer.domElement.addEventListener('mousemove',e=>{{const dx=e.clientX-prev.x,dy=e.clientY-prev.y;if(isDrag){{sph.theta-=dx*0.005;sph.phi=Math.max(0.1,Math.min(Math.PI-0.1,sph.phi-dy*0.005));updateCam();}}if(isRight){{const right=new THREE.Vector3().crossVectors(camera.up,camera.getWorldDirection(new THREE.Vector3())).normalize();pan.addScaledVector(right,dx*sph.radius*0.001);pan.addScaledVector(camera.up,-dy*sph.radius*0.001);updateCam();}}prev={{x:e.clientX,y:e.clientY}};}});
renderer.domElement.addEventListener('mouseup',()=>{{isDrag=false;isRight=false;}});
renderer.domElement.addEventListener('wheel',e=>{{sph.radius=Math.max(5,Math.min(extent*5,sph.radius*(1+e.deltaY*0.001)));updateCam();}});
renderer.domElement.addEventListener('contextmenu',e=>e.preventDefault());
document.getElementById('opacitySlider').addEventListener('input',e=>{{const v=e.target.value/100;document.getElementById('opacityVal').textContent=v.toFixed(2);if(vesselMesh)vesselMesh.material.opacity=v;}});
document.getElementById('widthSlider').addEventListener('input',e=>{{const v=parseInt(e.target.value);document.getElementById('widthVal').textContent=v;edgeMats.forEach(m=>m.linewidth=v);}});
document.getElementById('nodeSizeSlider').addEventListener('input',e=>{{const v=parseInt(e.target.value)/10;document.getElementById('nodeSizeVal').textContent=v.toFixed(1);nodeGroup.children.forEach(c=>{{if(c.isMesh)c.scale.setScalar(v);}});}});
document.getElementById('labelToggle').addEventListener('change',e=>{{labelSprites.forEach(s=>s.visible=e.target.checked);}});
let html='<div class="legend-item"><span class="dot" style="background:#ff3333"></span> BIF</div>';
html+='<div class="legend-item"><span class="dot" style="background:#3399ff"></span> EP</div>';
html+='<hr style="border-color:#223;margin:6px 0">';
EDGES.forEach((e,i)=>{{html+=`<div class="legend-item"><span class="dot" style="background:${{e.color||PAL[i%PAL.length]}}"></span> ${{e.label}}</div>`;}});
document.getElementById('legendContainer').innerHTML=html;
(function animate(){{requestAnimationFrame(animate);renderer.render(scene,camera);}})();
window.addEventListener('resize',()=>{{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);}});
</script></body></html>"""
    with open(out_path, 'w') as f: f.write(html)
    print(f"  HTML saved: {out_path}")


# ── public API ────────────────────────────────────────────────────────────────

def build_graph(planb_output_dir, out_dir, n_phases=25, skip_segments=None,
                bif_cluster_radius=3, ep_cluster_radius=2, mask_downsample=2):
    """Build vessel graph from Plan B segments, save PKL + NIfTI overlays + HTML."""
    if skip_segments is None: skip_segments = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(planb_output_dir, "segments.json")) as f:
        seg_data = json.load(f)
    all_segments = {int(k): np.array(v) for k, v in seg_data.items()}
    segments = {sid: c for sid, c in all_segments.items() if sid not in skip_segments}
    print(f"  Segments: {len(all_segments)} total, {len(segments)} active")

    p1_img     = nib.load(os.path.join(planb_output_dir, "p01_binary.nii.gz"))
    mask_shape = p1_img.shape
    affine     = p1_img.affine
    vox_size   = np.abs(np.diag(affine)[:3])

    # Reconstruct skeleton
    skel = np.zeros(mask_shape, dtype=np.uint8)
    for sid, coords in segments.items():
        skel[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

    eps_raw, bifs_raw, _ = _classify_voxels(skel)
    bif_clusters = _cluster_voxels(bifs_raw, bif_cluster_radius)
    ep_clusters  = _cluster_voxels(eps_raw,  ep_cluster_radius)
    bifs = [r for r, _ in bif_clusters]
    eps  = [r for r, _ in ep_clusters]
    print(f"  Nodes: bif={len(bifs)}, ep={len(eps)}")

    node_ids, edges = _extract_graph(skel, bif_clusters, ep_clusters, vox_size)
    print(f"  Edges: {len(edges)}")
    for key, e in sorted(edges.items(), key=lambda x: x[1]['node_ids']):
        print(f"    {e['node_ids'][0]:10s} — {e['node_ids'][1]:10s}  {e['length_mm']:.1f} mm")

    # Save reference graph
    ref_data = {'source': 'planb_segments', 'affine': affine, 'voxel_size': vox_size,
                'shape': mask_shape, 'skel': skel, 'segments': segments,
                'bifs': bifs, 'eps': eps, 'node_ids': node_ids, 'edges': edges,
                'skip_segments': skip_segments}
    ref_path = os.path.join(out_dir, "reference_graph.pkl")
    with open(ref_path, 'wb') as f: pickle.dump(ref_data, f)
    print(f"  Graph saved: {ref_path}")

    # NIfTI overlays
    overlay = np.zeros(mask_shape, dtype=np.uint8)
    overlay[skel > 0] = 1
    for group, label in [(bifs, 2), (eps, 3)]:
        for v in group:
            for dz in range(-2,3):
                for dy in range(-2,3):
                    for dx in range(-2,3):
                        if dz*dz+dy*dy+dx*dx<=4:
                            zz,yy,xx = int(v[0])+dz,int(v[1])+dy,int(v[2])+dx
                            if all(0<=c<s for c,s in zip([zz,yy,xx],mask_shape)):
                                overlay[zz,yy,xx] = label
    nib.save(nib.Nifti1Image(overlay, affine),
             os.path.join(out_dir, "reference_skel_overlay.nii.gz"))
    skel_col = np.zeros(mask_shape, dtype=np.uint8)
    for sid, coords in segments.items():
        skel_col[coords[:,0],coords[:,1],coords[:,2]] = sid
    nib.save(nib.Nifti1Image(skel_col, affine),
             os.path.join(out_dir, "reference_segments.nii.gz"))

    # Per-phase PKL
    for ph in range(1, n_phases + 1):
        fname = os.path.join(planb_output_dir, f"p{ph:02d}_binary.nii.gz")
        if not os.path.exists(fname): continue
        img_ph  = nib.load(fname)
        mask_ph = img_ph.get_fdata().astype(bool)
        with open(os.path.join(out_dir, f"phase_{ph:02d}_data.pkl"), 'wb') as f:
            pickle.dump({'phase': ph, 'affine': img_ph.affine,
                         'voxel_size': np.abs(np.diag(img_ph.affine)[:3]),
                         'shape': mask_ph.shape, 'mask': mask_ph}, f)

    # 3D HTML
    nids = {tuple(int(x) for x in k): v for k, v in node_ids.items()}
    verts, faces = _surface_mesh(p1_img.get_fdata().astype(bool), affine, mask_downsample)
    nodes_vis = [{'pos': (affine @ np.array([*vox, 1.0]))[:3].tolist(),
                  'label': label, 'type': 'BIF' if label.startswith('BIF') else 'EP'}
                 for vox, label in nids.items()]
    PAL = ['#ff6b6b','#4ecdc4','#45b7d1','#96ceb4','#ffeaa7','#dfe6e9','#fd79a8','#a29bfe',
           '#00b894','#e17055','#74b9ff','#55efc4','#fab1a0','#81ecec','#b2bec3','#6c5ce7']
    edge_paths = []
    for i, (key, e) in enumerate(sorted(edges.items(), key=lambda x: x[1]['node_ids'])):
        step   = max(1, len(e['path']) // 100)
        sampled = e['path'][::step]
        if e['path'][-1] not in sampled: sampled.append(e['path'][-1])
        pts_mm = [(affine @ np.array([*v, 1.0]))[:3].tolist() for v in sampled]
        edge_paths.append({'points': pts_mm,
                           'label': f"{e['node_ids'][0]} — {e['node_ids'][1]} ({e['length_mm']:.1f}mm)",
                           'color': PAL[i % len(PAL)]})
    html_path = os.path.join(out_dir, "vessel_graph_3d.html")
    _generate_html(verts, faces, nodes_vis, edge_paths, html_path)
    print("  Build graph complete.")