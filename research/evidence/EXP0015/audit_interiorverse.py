"""Verify the endorsed InteriorVerse shard and inspect bounded material samples.

This dataset is synthetic. A depth/normal image is not a complete scene asset.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(root, out, count):
    out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((root/'dataset_85_tree.json').read_text())
    entry=next(x for x in metadata if x['path']=='dataset_85/part_0.zip')
    archive=root/'part_0.zip'
    if not archive.exists():archive=root/'part_0.zip.partial'
    actual=sha(archive)
    if archive.stat().st_size!=entry['size'] or actual!=entry['lfs']['oid']:
        raise ValueError('Archive length or authoritative LFS SHA-256 mismatch')
    if archive.suffix=='.partial':archive=archive.rename(root/'part_0.zip')
    splits={k:set((root/(k+'.txt')).read_text().split()) for k in ['train','val','test']}
    assert not (splits['train']&splits['test'] or splits['train']&splits['val'] or splits['test']&splits['val'])
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad is not None:raise ValueError('CRC failure '+bad)
        files=[i for i in z.infolist() if not i.is_dir()]
        inventory=[dict(path=i.filename,bytes=i.file_size,crc32=i.CRC) for i in files]
        (out/'archive_inventory.json').write_text(json.dumps(inventory,indent=2))
        scenes=sorted({Path(i.filename).parts[0] for i in files})
        selected=scenes[:count]
        records=[];rows=[]
        for s in selected:
            scene_files=[i for i in files if Path(i.filename).parts[0]==s]
            for i in scene_files:
                dest=(root/'sample'/i.filename).resolve()
                if not dest.is_relative_to((root/'sample').resolve()):raise ValueError('Unsafe ZIP path')
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(i) as src,dest.open('wb') as target:shutil.copyfileobj(src,target)
            folder=root/'sample'/s
            views=sorted(folder.glob('*_im.exr'))
            for index,p in enumerate(views):
                arrays={}
                for modality in ['im','mask','albedo','depth','material','normal']:
                    path=folder/(p.name.replace('_im.exr','_'+modality+'.exr'))
                    a=cv2.imread(str(path),-1)
                    if a is None:raise ValueError('Unreadable '+str(path))
                    if a.ndim==3:a=a[...,::-1]
                    arrays[modality]=a
                mask=arrays['mask']
                if mask.ndim==3:mask=mask[...,0]
                valid=mask>.5
                if not valid.any():raise ValueError('Empty mask')
                for k,a in arrays.items():
                    if not np.isfinite(a[valid]).all():raise ValueError('Invalid valid-pixel value: '+str(p)+' '+k)
                mat=arrays['material']
                if mat[valid,:2].min()<-.001 or mat[valid,:2].max()>1.001:raise ValueError('Material bounds')
                records.append(dict(scene=s,view=p.name,split=next((k for k,v in splits.items() if s in v),None),
                    resolution=list(mask.shape),valid_fraction=float(valid.mean()),
                    roughness_range=[float(mat[...,0][valid].min()),float(mat[...,0][valid].max())],
                    metallic_range=[float(mat[...,1][valid].min()),float(mat[...,1][valid].max())],
                    hdr_max=float(arrays['im'][valid].max())))
                if index==0:
                    row=[]
                    panels=[('HDR preview',np.clip(arrays['im'],0,1)**(1/2.2)),
                            ('Albedo',np.clip(arrays['albedo'],0,1)**(1/2.2)),
                            ('Roughness',np.repeat(mat[...,:1],3,-1)),
                            ('Metallic',np.repeat(mat[...,1:2],3,-1)),
                            ('Normals',(arrays['normal']+1)/2)]
                    for label,a in panels:
                        img=np.uint8(np.clip(np.nan_to_num(a)*255,0,255))[...,::-1]
                        img=cv2.resize(img,(256,192))
                        cv2.putText(img,label,(6,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                        row.append(img)
                    rows.append(np.hstack(row))
        cv2.imwrite(str(out/'material_samples.jpg'),np.vstack(rows))
    other=[x['path'] for x in inventory if not x['path'].endswith('.exr')]
    result=dict(provenance='synthetic_artist_authored',original_url_status=404,
        endorsed_mirror='https://huggingface.co/datasets/Lez/InteriorVerse',
        archive='dataset_85/part_0.zip',archive_bytes=entry['size'],sha256=actual,
        sha256_matches_hf_lfs=True,all_zip_crcs_pass=True,archive_scenes=len(scenes),
        archive_files=len(files),file_types=dict(Counter(Path(x['path']).suffix for x in inventory)),
        non_exr_files=other,sampled_scenes=selected,sampled_views=len(records),
        material_channels='RGB: roughness, metallic, unused; GGX convention needs model mapping',
        depth_unit='millimetres; invalid depth may be inf',normal_frame='OpenGL camera',
        explicit_sun_parameters_found=False,complete_scene_assets_found=False,
        lighting_release='official README still marks spatially-varying lighting unreleased',
        role='material-map evaluation candidate; not real scans or verified solar rerender dataset',
        records=records)
    (out/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('data_download/interiorverse'))
    p.add_argument('--out',type=Path,default=Path('experiments/out/EXP0015_interiorverse'))
    p.add_argument('--scenes',type=int,default=3)
    a=p.parse_args();run(a.root,a.out,a.scenes)
