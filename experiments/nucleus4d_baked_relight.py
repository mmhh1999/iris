"""Offline mesh lighting transferred onto the unchanged full Nucleus4D splat.

This is a local, diffuse living-room experiment inspired by the public
mesh-to-Gaussian workflow. It is not Niantic's unpublished implementation.
Uncovered rooms and uncertain/specular surfaces retain their original colors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/out/nucleus4d_baked_relight'
SOURCE = ROOT / 'experiments/out/nucleus4d_3dgs'
MATERIAL = ROOT / 'experiments/out/nucleus4d_material'
PTIR = ROOT / 'experiments/out/nucleus4d_ptir'
PHOTOS = ROOT / 'experiments/out/nucleus4d_ptir_full'
SH_C0 = .28209479177387814


def atomic_json(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix('.partial.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)


def file_identity(path):
    p = Path(path).resolve(); s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)


def srgb_to_linear(value):
    value = np.asarray(value, dtype=np.float32)
    return np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError('Radiance must be finite and nonnegative')
    return np.where(value <= .0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - .055)


def geometry_digest(rows):
    """All geometry, ordering and opacity bytes; only RGB may change."""
    digest = hashlib.sha256()
    for start in range(0, len(rows), 100000):
        block = rows[start:start+100000]
        digest.update(block[:, :24].tobytes()); digest.update(block[:, 27:].tobytes())
    return digest.hexdigest()


def transfer_diffuse(albedo, mesh_radiance, mesh_albedo):
    if (not all(np.isfinite(a).all() for a in (albedo, mesh_radiance, mesh_albedo))
            or (albedo < 0).any() or (mesh_radiance < 0).any() or (mesh_albedo <= 0).any()):
        raise ValueError('Invalid material or mesh radiance')
    # Lambertian outgoing radiance = rho * irradiance / pi. Division removes
    # only the proxy's first-hit albedo; each Gaussian keeps its own texture.
    return (albedo * mesh_radiance / mesh_albedo).astype(np.float32)


def read_source_ply(path):
    with Path(path).open('rb') as stream:
        header = bytearray(); count = None; names = []
        while True:
            line = stream.readline()
            if not line: raise ValueError('Truncated PLY header')
            header.extend(line)
            if line.startswith(b'element vertex '): count = int(line.split()[-1])
            if line.startswith(b'property float '): names.append(line.split()[-1].decode())
            if line.strip() == b'end_header': break
    expected = ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity',
                'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    if b'format binary_little_endian 1.0' not in header or names != expected:
        raise ValueError('Use the unmodified Nucleus SH0 source; never discard higher SH bands')
    array = np.memmap(path, mode='r', dtype='<f4', offset=len(header), shape=(count, len(names)))
    return bytes(header), array


def prepare(output, progress):
    from scipy.spatial import cKDTree
    inputs = [SOURCE/'full.splat', PHOTOS/'source.cache.ply', MATERIAL/'surface.npz',
              MATERIAL/'surface_materials.npz', PTIR/'materials.npz',
              PHOTOS/'indoor_photo_repair_20261021/indoor_lights.json', MATERIAL/'views.json',
              Path(__file__), Path(__file__).with_name('nucleus4d_mesh_bake.py')]
    identity = [file_identity(p) for p in inputs]
    plan_path = output/'plan.json'; cache = output/'mapping.npz'
    if plan_path.exists() and cache.exists():
        plan = json.loads(plan_path.read_text())
        if plan['inputs'] != identity: raise RuntimeError('Inputs changed; use a fresh output directory')
        return plan
    raw = np.memmap(SOURCE/'full.splat', mode='r', dtype='u1').reshape(-1,32)
    _, source = read_source_ply(PHOTOS/'source.cache.ply')
    if len(raw) != len(source): raise RuntimeError('Full viewer and original PLY counts differ')
    positions = raw[:, :12].copy().view('<f4').reshape(-1,3)
    with np.load(MATERIAL/'surface.npz') as a: vertices = a['vertices'].copy()
    with np.load(MATERIAL/'surface_materials.npz') as a: confidence = a['confidence'].copy()
    with np.load(PTIR/'materials.npz') as a:
        materials = {k:a[k].astype(np.float32) for k in ('albedo','roughness','metallic','support')}
    if any(len(a) != len(source) for a in materials.values()): raise ValueError('Material order/count mismatch')
    lower, upper = vertices.min(0)-.12, vertices.max(0)+.12
    source_ids = np.flatnonzero(((source[:,:3]>=lower)&(source[:,:3]<=upper)).all(1))
    tree = cKDTree(np.asarray(source[source_ids,:3])); surface_tree = cKDTree(vertices)
    eligible = np.flatnonzero(((positions>=lower)&(positions<=upper)).all(1))
    mapped = np.full(len(raw), -1, np.int32); source_index = mapped.copy()
    rejected_ambiguous = 0
    for start in range(0,len(eligible),100000):
        ids = eligible[start:start+100000]
        distance, nearest = tree.query(positions[ids], k=2, workers=4)
        exact = (distance[:,0]<1e-7)&(distance[:,1]>1e-7)
        rejected_ambiguous += int((distance[:,1]<=1e-7).sum())
        original_ids = source_ids[nearest[:,0]]
        d, v = surface_tree.query(positions[ids], workers=4)
        roughness = materials['roughness'][original_ids].reshape(-1)
        metallic = materials['metallic'][original_ids].reshape(-1)
        support = materials['support'][original_ids].reshape(-1)
        # Learned priors assign intermediate metallic values even to room walls.
        # Use an explicit diffuse assumption ONLY for the known enclosing shell;
        # do not quietly interpret those uncertain priors as measured metals.
        q=positions[ids]
        shell=(np.abs(q[:,2]+1.18)<.10)|(np.abs(q[:,2]-1.38)<.10)
        shell|=(np.abs(q[:,0]+1.80)<.10)|(np.abs(q[:,0]-5.98)<.10)
        shell|=(np.abs(q[:,1]+2.50)<.10)|(np.abs(q[:,1]-2.45)<.10)
        diffuse=shell|((support>.1)&(roughness>=.35)&(metallic<.2))
        valid = exact & (d<.08) & (confidence[v]>=.333) & diffuse
        mapped[ids[valid]] = v[valid]; source_index[ids[valid]] = original_ids[valid]
        progress('mapping', mapped=int((mapped>=0).sum()), checked=min(start+len(ids),len(eligible)), total=len(eligible))
    used = np.unique(mapped[mapped>=0])
    if len(used)==0: raise RuntimeError('No confidently supported diffuse surface matches')
    atomic_npz(cache, node=mapped, source_index=source_index, used_vertices=used)
    plan = dict(schema=1, inputs=identity, gaussians=len(raw), source_sh_degree=0,
                geometry_sha256=geometry_digest(raw), mapped_gaussians=int((mapped>=0).sum()),
                mapped_fraction=float((mapped>=0).mean()), nodes=len(used),
                rejected_coincident_ambiguous=rejected_ambiguous, spp=1024, bounces=8,
                dtype='float32', filtering='none', scope='supported diffuse living-room surfaces only',
                source_uncalibrated=True, physically_validated=False, original_modified=False,
                limitations=['Other rooms and unsupported/glossy surfaces retain captured illumination',
                 'Proxy geometry and open windows are estimates; glass is not recovered',
                 'Photo albedo and fixture powers are estimates, not measurements',
                 'Room enclosure is explicitly approximated as diffuse; learned metallic priors are not trusted there',
                 'SH0 source retains no view-dependent SH; glossy relighting is not established',
                 'This is fixed-light radiance baking, not a reproduction of Niantic PRT'],
                sources=dict(workflow='https://www.nianticspatial.com/blog/introducing-gaussian-splat-relighting'))
    atomic_json(plan_path,plan)
    return plan


def export(output, plan, field, progress):
    with np.load(output/'mapping.npz') as a:
        mapped=a['node']; source_index=a['source_index']; used=a['used_vertices']
    with np.load(MATERIAL/'surface_materials.npz') as a: proxy_albedo=a['albedo']
    with np.load(PTIR/'materials.npz') as a: albedo=a['albedo'].astype(np.float32)
    original=np.memmap(SOURCE/'full.splat',mode='r',dtype='u1').reshape(-1,32)
    partial=output/'relit.partial.splat'
    shutil.copyfile(SOURCE/'full.splat',partial)
    target=np.memmap(partial,mode='r+',dtype='u1').reshape(-1,32)
    hdr_path=output/'radiance.partial.npy'
    hdr=np.lib.format.open_memmap(hdr_path,mode='w+',dtype='<f4',shape=(len(original),3))
    lookup=np.full(len(proxy_albedo),-1,np.int32);lookup[used]=np.arange(len(used))
    changed=0;clipped=0;valid_count=0
    for start in range(0,len(original),100000):
        stop=min(start+100000,len(original)); nodes=mapped[start:stop]
        base=srgb_to_linear(original[start:stop,24:27].astype(np.float32)/255)
        ok=nodes>=0; indices=np.flatnonzero(ok); local=lookup[nodes[ok]]
        finite=np.isfinite(field[local]).all(1)&(field[local]>=0).all(1)
        indices=indices[finite];local=local[finite]
        transferred=transfer_diffuse(albedo[source_index[start:stop][indices]],field[local],proxy_albedo[nodes[indices]])
        base[indices]=transferred;hdr[start:stop]=base
        encoded=linear_to_srgb(transferred)
        clipped+=int((encoded>1).any(1).sum());valid_count+=len(indices)
        target[start+indices,24:27]=np.rint(encoded.clip(0,1)*255).astype('u1')
        changed+=int((target[start:stop,24:27]!=original[start:stop,24:27]).any(1).sum())
    target.flush();hdr.flush();del target,hdr
    baked=np.memmap(partial,mode='r',dtype='u1').reshape(-1,32)
    if geometry_digest(baked)!=plan['geometry_sha256']:raise RuntimeError('Geometry or opacity changed')
    unmapped=mapped<0
    if not np.array_equal(baked[unmapped],original[unmapped]):raise RuntimeError('Uncovered surfaces changed')
    partial.replace(output/'relit.splat');hdr_path.replace(output/'radiance.npy')
    # Float32 master PLY retains source geometry, opacity and radiance beyond 1.
    header, source=read_source_ply(PHOTOS/'source.cache.ply')
    ply_path=output/'relit.partial.ply'
    with ply_path.open('wb') as stream:stream.write(header);stream.truncate(len(header)+source.nbytes)
    master=np.memmap(ply_path,mode='r+',dtype='<f4',offset=len(header),shape=source.shape)
    for start in range(0,len(source),100000):master[start:start+100000]=source[start:start+100000]
    radiation=np.load(output/'radiance.npy',mmap_mode='r')
    for start in range(0,len(original),100000):
        stop=min(start+100000,len(original));ok=(source_index[start:stop]>=0)&(mapped[start:stop]>=0)
        ids=np.flatnonzero(ok)
        if not len(ids):continue
        local=lookup[mapped[start:stop][ids]];valid=np.isfinite(field[local]).all(1)
        ids=ids[valid]
        master[source_index[start:stop][ids],6:9]=(linear_to_srgb(radiation[start+ids])-.5)/SH_C0
    master.flush();del master
    ply_path.replace(output/'relit.ply')
    report=dict(status='experimental_partial', gaussians=len(original), changed_gaussians=changed,
                lighting_transferred_gaussians=valid_count, invalid_surface_nodes=int((~np.isfinite(field).all(1)).sum()),
                geometry_sha256=plan['geometry_sha256'], geometry_bitwise_unchanged=True,
                uncovered_bitwise_unchanged=True, source_sh_degree=0, master='relit.ply',
                master_dtype='float32', hdr='radiance.npy', hdr_unclipped=True,
                browser='relit.splat', browser_encoding='same uint8 RGB format as original viewer',
                browser_saturated_gaussians=clipped, appearance_passed=False,
                physical_accuracy_verified=False, original_quality_verified=False,
                scope=plan['scope'], limitations=plan['limitations'],completed_at=time.time())
    atomic_json(output/'result.json',report);progress('exported',**report)


def publish(output, plan):
    from nucleus4d_baked_relight_viewer import publish_viewer
    publish_viewer(output,plan,SOURCE,MATERIAL)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    def progress(stage,**values):
        record=dict(status='running',stage=stage,pid=os.getpid(),updated=time.time())
        record.update(values)
        atomic_json(output/'status.json',record);print('BAKE',json.dumps(record),flush=True)
    plan=prepare(output,progress)
    if args.prepare_only:progress('prepared',status='prepared',nodes=plan['nodes']);return
    from nucleus4d_mesh_bake import bake
    field=bake(output,plan,progress,threads=args.threads)
    export(output,plan,field,progress);publish(output,plan)
    progress('completed',status='experimental_partial',url='http://localhost:8769/'+output.name+'/',
             appearance_passed=False,physical_accuracy_verified=False)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        # No dead worker should remain advertised as running.
        output=Path(sys.argv[sys.argv.index('--output')+1]).resolve() if '--output' in sys.argv else OUT
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(output/'status.json',dict(status='needs_attention',error=f'{type(exc).__name__}: {exc}',updated=time.time()))
        raise
