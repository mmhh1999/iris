"""Kitchen geometry, fixed IRIS-matched matte floor, moving direct sunlight."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import argparse,json,hashlib,shutil,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')

class IrisMatte(mi.BSDF):
    def __init__(self,p):
        mi.BSDF.__init__(self,p)
        self.m_flags=mi.BSDFFlags.DiffuseReflection|mi.BSDFFlags.FrontSide
        self.m_components=[self.m_flags]
    def eval(self,ctx,si,wo,active=True):
        ci=si.wi.z;co=wo.z;h=dr.normalize(si.wi+wo)
        r=.9;aa=r**4;D=aa/(dr.pi*dr.square(h.z*h.z*(aa-1)+1))
        k=(r+1)**2/8;G=1/((ci*(1-k)+k)*(co*(1-k)+k))
        F=.04+.96*(1-dr.dot(wo,h))**5
        value=mi.Color3f(.28,.20,.12)/dr.pi*co+D*G*F/4*co
        return dr.select(active&(ci>0)&(co>0),value,0)
    def pdf(self,ctx,si,wo,active=True):return dr.select(active&(si.wi.z>0),dr.maximum(wo.z,0)/dr.pi,0)
    def sample(self,ctx,si,sample1,sample2,active=True):
        bs=mi.BSDFSample3f();bs.wo=mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf=self.pdf(ctx,si,bs.wo,active);bs.eta=1;bs.sampled_type=mi.UInt32(mi.BSDFFlags.DiffuseReflection);bs.sampled_component=0
        return bs,self.eval(ctx,si,bs.wo,active)/dr.maximum(bs.pdf,1e-8)
    def eval_pdf(self,ctx,si,wo,active=True):return self.eval(ctx,si,wo,active),self.pdf(ctx,si,wo,active)
    def to_string(self):return 'IRIS matched matte floor (albedo .28 .20 .12, roughness .9, metallic 0)'
mi.register_bsdf('iris_matte',lambda p:IrisMatte(p))

def exr(p,a):
    p.parent.mkdir(parents=True,exist_ok=True);a=np.asarray(a,dtype=np.float32)
    if not np.isfinite(a).all():raise ValueError(p)
    if not cv2.imwrite(str(p),a[...,::-1]):raise IOError(p)

def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
    original=Path('data_download/pbr_interiors/kitchen/kitchen/scene_v3.xml').resolve()
    root=ET.parse(original).getroot()
    for s in list(root.findall('shape')):
        if s.find('emitter') is not None:root.remove(s)
    for e in root.iter('string'):
        if e.get('name')=='filename':e.set('value',str(original.parent/e.get('value')))
    floor=next(e for e in root.iter('bsdf') if e.get('id')=='FloorBSDF');floor.clear();floor.set('id','FloorBSDF');floor.set('type','iris_matte')
    # A small, fixed ceiling light supplies an observable indoor emitter for IRIS.
    lamp=ET.SubElement(root,'shape',type='rectangle',id='fixed_ceiling_lamp')
    tr=ET.SubElement(lamp,'transform',name='to_world');ET.SubElement(tr,'rotate',x='1',angle='90');ET.SubElement(tr,'scale',x='.25',y='.25',z='1');ET.SubElement(tr,'translate',x='0',y='3.1',z='1.5')
    em=ET.SubElement(lamp,'emitter',type='area');ET.SubElement(em,'rgb',name='radiance',value='15,15,15')
    sky=ET.SubElement(root,'emitter',type='constant');ET.SubElement(sky,'rgb',name='radiance',value='.12,.14,.18')
    sensor=root.find('sensor');sensor.find('float[@name="fov"]').set('value','65')
    for e in root.findall('default'):
        vals={'resx':a.width,'resy':a.height,'max_depth':8,'spp':a.spp}
        if e.get('name') in vals:e.set('value',str(vals[e.get('name')]))
    # Cameras share floor coverage while varying lateral location and height.
    poses=[]
    for i in range(12):
        x=-.5+(i%4)*.5;z=2.8+(i//4)*.5;y=1.5+.15*(i%3)
        pose=mi.ScalarTransform4f.look_at(origin=[x,y,z],target=[.1,.15,-.8],up=[0,1,0]);poses.append(np.array(pose.matrix).tolist())
    # Two solar conditions for independent fits, one unseen condition and no-sun control.
    conditions=[('sun_a',-25,25),('sun_b',25,25),('sun_c',0,40),('sky',0,0)]
    manifest={'experiment':'EXP0019','scene':'Country Kitchen authored geometry; modified matte floor and fixed ceiling light',
       'license':'Country Kitchen by Jay-Artist CC BY 3.0, curated by Benedikt Bitterli https://noobody.org/resources/',
       'floor_albedo':[.28,.20,.12],'floor_roughness':.9,'floor_metallic':0,'spp':a.spp,
       'resolution':[a.width,a.height],'sun_irradiance':[4,3.7,3.2],'sun_model':'directional zero angular radius','conditions':[],
       'protocol':'known geometry and camera diagnostic (P1), not photo-only reconstruction',
       'prior':'neutral constant .5 albedo image; not original Irisformer prior','fit_conditions':['sun_a','sun_b','sky'],'heldout_light':'sun_c'}
    gt=out/'generator_truth';gt.mkdir()
    for name,az,el in conditions:
        croot=ET.fromstring(ET.tostring(root));d=None
        if name!='sky':
            azr,elr=np.deg2rad([az,el]);d=np.array([np.sin(azr)*np.cos(elr),np.sin(elr),-np.cos(azr)*np.cos(elr)])
            sun=ET.SubElement(croot,'emitter',type='directional');ET.SubElement(sun,'vector',name='direction',value=','.join(map(str,-d)));ET.SubElement(sun,'rgb',name='irradiance',value='4,3.7,3.2')
        rows=[];datasets=out/name
        for i,pose in enumerate(poses):
            split='val' if i in [3,9] else 'train';idx=sum(1 for r in rows if r['split']==split)
            croot.find('sensor/transform/matrix').set('value',' '.join(map(str,np.array(pose).ravel())))
            xml=gt/f'{name}_view{i:02d}.xml';ET.ElementTree(croot).write(xml)
            scene=mi.load_file(str(xml.resolve()));im=np.array(mi.render(scene,spp=a.spp,seed=101))
            folder=datasets/split;exr(folder/'Image'/f'{idx:03d}_0001.exr',im)
            cv2.imwrite(str(folder/'Image'/f'{idx:03d}_0001.png'),np.uint8(np.clip(im,0,1)**(1/2.2)*255)[...,::-1])
            yy,xx=np.mgrid[:a.height,:a.width];ray,_=scene.sensors()[0].sample_ray(0.,0.,mi.Point2f(mi.Float(((xx+.5)/a.width).ravel()),mi.Float(((yy+.5)/a.height).ravel())),mi.Point2f(.5,.5));si=scene.ray_intersect(ray)
            fl=next(s for s in scene.shapes() if s.id()=='Floor');mask=np.asarray(si.shape==fl).reshape(a.height,a.width)
            patch=np.zeros_like(mask)
            if d is not None:
                shadow=mi.Ray3f(si.p+mi.Vector3f(*map(float,d))*.0001,mi.Vector3f(*map(float,d)))
                patch=mask & ~np.asarray(scene.ray_test(shadow)).reshape(mask.shape)
            np.savez_compressed(gt/f'{name}_view{i:02d}.npz',floor=mask,sun_visible=patch,position=np.array(si.p).T.reshape(a.height,a.width,3))
            # GT fields required by stock loader: accurate only on evaluation floor.
            zeros=np.zeros_like(im);alb=zeros.copy();alb[mask]=[.28,.20,.12];rough=zeros.copy();rough[mask]=.9
            for field,v in [('DiffCol',alb),('Roughness',rough),('Emit',zeros),('IndexMA',zeros)]:exr(folder/field/f'{idx:03d}_0001.exr',v)
            prior=folder/'Image/albedo';prior.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(prior/f'{idx:03d}_0001.png'),np.full(im.shape,128,np.uint8))
            rows.append({'view':i,'split':split,'index':idx,'transform_matrix':pose,'floor_pixels':int(mask.sum()),'sun_pixels':int(patch.sum()),'floor_saturation_fraction':float((im[mask]>=1).any(-1).mean())})
            if name=='sun_a' and i==0:
                # Export exact geometric triangles to the stock IRIS mesh format.
                vs=[];fs=[];count=0
                for shape in scene.shapes():
                    p=mi.traverse(shape)
                    if 'vertex_positions' not in p:raise ValueError(f'Non-mesh shape {shape.id()}')
                    v=np.array(p['vertex_positions']).reshape(-1,3);f=np.array(p['faces']).reshape(-1,3)
                    vs.append(v);fs.append(f+count);count+=len(v)
                with (out/'scene.obj').open('w') as f:
                    for v in np.concatenate(vs):f.write('v '+' '.join(map(str,v))+'\n')
                    for q in np.concatenate(fs)+1:f.write('f '+' '.join(map(str,q))+'\n')
            print(name,i,'floor',int(mask.sum()),'sun',int(patch.sum()),flush=True)
        for split in ['train','val']:
            rr=[r for r in rows if r['split']==split];folder=datasets/split
            (folder/'transforms.json').write_text(json.dumps({'camera_angle_x':float(np.deg2rad(65)),'frames':rr},indent=2))
            cam=folder/'Image/cam';cam.mkdir();np.save(cam/'exposure.npy',np.ones(len(rr),np.float32));np.save(cam/'crf.npy',np.tile(np.linspace(0,1,1024)**(1/2.2),(3,1)).astype(np.float32))
        shutil.copy2(out/'scene.obj',datasets/'scene.obj')
        manifest['conditions'].append({'name':name,'sun_toward_world':None if d is None else d.tolist(),'views':rows})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='experiments/out/EXP0019_sunpatch/data_01');p.add_argument('--width',type=int,default=128);p.add_argument('--height',type=int,default=96);p.add_argument('--spp',type=int,default=1024);run(p.parse_args())
