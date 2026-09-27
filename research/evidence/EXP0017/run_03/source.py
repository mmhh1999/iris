"""Textured authored-room solar controls. Preserve original BSDFs and assets.

Stage A is renderer/data validation, not a real scan or inverse BRDF result.
Directional sun is a zero-angular-size approximation; sky is uniform RGB.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import mitsuba as mi


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_image(path,array):
    a=np.asarray(array,dtype=np.float32)
    if not np.isfinite(a).all():raise ValueError('Nonfinite render')
    if not cv2.imwrite(str(path.with_suffix('.exr')),a[...,::-1]):raise IOError(path)
    # Same fixed exposure and display curve for every camera/time/seed.
    cv2.imwrite(str(path.with_suffix('.png')),np.uint8(np.clip(a,0,1)**(1/2.2)*255)[...,::-1])
    return a


def build(source,out,name,direction,offset,width,height):
    tree=ET.parse(source);root=tree.getroot();removed=[]
    for shape in list(root.findall('shape')):
        if shape.find('emitter') is not None:
            removed.append(ET.tostring(shape,encoding='unicode'));root.remove(shape)
    for e in list(root.findall('emitter')):root.remove(e)
    for e in root.iter('string'):
        if e.get('name')=='filename':e.set('value',str((source.parent/e.get('value')).resolve()))
    sensor=root.find('sensor');matrix=sensor.find('transform/matrix')
    pose=np.array([float(x) for x in matrix.get('value').split()]).reshape(4,4)
    # The original bathroom camera sits near a wall; lateral translation can
    # leave the room. Move forward along its optical axis instead.
    pose[:3,3]+=pose[:3,2]*offset
    matrix.set('value',' '.join(str(x) for x in pose.ravel()))
    defaults={'resx':width,'resy':height,'max_depth':12,'spp':64}
    for e in root.findall('default'):
        if e.get('name') in defaults:e.set('value',str(defaults[e.get('name')]))
    sky=ET.SubElement(root,'emitter',type='constant',id='pilot_sky')
    ET.SubElement(sky,'rgb',name='radiance',value='0.12,0.14,0.18')
    if direction is not None:
        sun=ET.SubElement(root,'emitter',type='directional',id='pilot_sun')
        ET.SubElement(sun,'vector',name='direction',value=','.join(str(-x) for x in direction))
        ET.SubElement(sun,'rgb',name='irradiance',value='4.0,3.7,3.2')
    xml=out/(name+'.xml');tree.write(xml,encoding='unicode')
    return xml,removed,pose.tolist()


def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'source.py');mi.set_variant('cuda_ad_rgb')
    record={'experiment':'EXP0017','source':'artist-authored PBR rooms, not real scans',
        'sun':'directional, zero angular diameter approximation','sky':'constant RGB radiance',
        'spp':a.spp,'depth':12,'seed_pair':[101,202],'source_sha256':sha(__file__),
        'mitsuba':mi.__version__,'observations':[],'rooms':{},'brdf_estimation_performed':False}
    conditions=[(-20,25),(0,40),(20,55)]
    begin=time.monotonic()
    for room in a.scenes:
        source=Path('data_download/pbr_interiors')/room/room/'scene_v3.xml'
        folder=out/room;folder.mkdir();xmlroot=ET.parse(source).getroot()
        first=next(s for s in xmlroot.findall('shape') if s.find('emitter') is not None)
        transform=np.array([float(x) for x in first.find('transform/matrix').get('value').split()]).reshape(4,4)
        up=np.array([0.,1.,0.]);outward=-transform[:3,2];outward[1]=0;outward/=np.linalg.norm(outward)
        tangent=np.cross(up,outward)
        record['rooms'][room]={'original_xml_sha256':sha(source),'license':(source.parent/'LICENSE.txt').read_text(),
                               'outward_axis':outward.tolist(),'modified_assets':'Remove original emissive rectangle shapes; all remaining BSDFs and geometry retained'}
        for view,offset in enumerate([0.,.12]):
            for condition,(az,el) in enumerate(conditions):
                azr,elr=np.radians([az,el]);d=np.cos(elr)*(np.cos(azr)*outward+np.sin(azr)*tangent)+np.sin(elr)*up
                tag=f'view{view}_sun{condition}'
                xml,removed,pose=build(source,folder,tag,d,offset,a.width,a.height)
                if view==0 and condition==0:record['rooms'][room]['removed_shapes']=removed
                scene=mi.load_file(str(xml.resolve()))
                images=[]
                for seed in [101,202]:
                    im=write_image(folder/f'{tag}_seed{seed}',mi.render(scene,spp=a.spp,seed=seed));images.append(im)
                    row={'room':room,'view':view,'condition':condition,'seed':seed,'path':f'{room}/{tag}_seed{seed}.exr',
                         'sun_toward_world':d.tolist(),'elevation_deg':el,'azimuth_relative_to_aperture_deg':az,
                         'c2w_mitsuba':pose,'mean_rgb':float(im.mean()),'max_rgb':float(im.max()),
                         'display_clip_fraction':float((im>1).mean())}
                    record['observations'].append(row)
                    (out/'manifest.json').write_text(json.dumps(record,indent=2))
                    print('RENDER',room,view,condition,seed,round(time.monotonic()-begin,1),flush=True)
                row['paired_seed_rmse']=float(np.sqrt(np.mean((images[0]-images[1])**2)))
            # Sky-only intervention checks that sunlight enters the actual room.
            xml,_,_=build(source,folder,f'view{view}_sky_only',None,offset,a.width,a.height)
            write_image(folder/f'view{view}_sky_only',mi.render(mi.load_file(str(xml.resolve())),spp=a.spp,seed=101))
        record['elapsed_seconds']=time.monotonic()-begin
        (out/'manifest.json').write_text(json.dumps(record,indent=2))
    print('COMPLETE',len(record['observations']),'observations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='experiments/out/EXP0017_pbr_solar/run_01')
    p.add_argument('--scenes',nargs='+',default=['bathroom','kitchen'])
    p.add_argument('--spp',type=int,default=256);p.add_argument('--width',type=int,default=384);p.add_argument('--height',type=int,default=256)
    run(p.parse_args())
