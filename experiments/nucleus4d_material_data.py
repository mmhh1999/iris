import pathlib,tarfile,struct,json,numpy as np
from scipy.spatial.transform import Rotation
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/out/nucleus4d_material';OUT.mkdir(parents=True,exist_ok=True)
p=pathlib.Path('/mnt/e/Datasets/Nucleus4D/20260929/1406-C-int.tar')
with tarfile.open(p,'r:') as t:
 for m in t:
  if m.name.endswith('/cameras.bin'):
   f=t.extractfile(m);n=struct.unpack('<Q',f.read(8))[0];cams={}
   for _ in range(n):
    cid,model,w,h=struct.unpack('<iiQQ',f.read(24));num={0:3,1:4,2:4,3:5,4:8,5:8,6:12,7:5,8:4,9:5,10:12}[model];params=list(struct.unpack('<'+'d'*num,f.read(num*8)));cams[cid]=dict(id=cid,model=model,width=w,height=h,params=params)
   (OUT/'raw_cameras.json').write_text(json.dumps(cams,indent=2));print('cameras',cams,flush=True)
  if m.name.endswith('/images.bin'):
   f=t.extractfile(m);n=struct.unpack('<Q',f.read(8))[0];rows=[]
   for _ in range(n):
    fields=struct.unpack('<i7di',f.read(64));iid=fields[0];q=np.array(fields[1:5]);tr=np.array(fields[5:8]);cid=fields[8];name=bytearray()
    while True:
     c=f.read(1)
     if c==b'\0':break
     if not c:raise RuntimeError('truncated image record')
     name+=c
    npts=struct.unpack('<Q',f.read(8))[0];f.seek(npts*24,1)
    r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();center=-r.T@tr
    rows.append(dict(id=iid,camera_id=cid,name=name.decode(),qvec=q.tolist(),tvec=tr.tolist(),position=center.tolist(),rotation=r.T.tolist()))
   (OUT/'raw_images.json').write_text(json.dumps(rows,indent=2));pos=np.array([r['position'] for r in rows]);print('images',n,'bounds',np.quantile(pos,[0,.5,1],axis=0),flush=True);break
