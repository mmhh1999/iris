"""Independent check against original COLMAP extrinsics; no image fitting."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.geometry_screening import scannetpp_pose_in_mesh_frame

def validate(root, scene_ids):
    # Load standalone existing reader without dataset/__init__ GPU dependencies.
    spec=importlib.util.spec_from_file_location('colmap_reader',ROOT/'utils/dataset/scannetpp/colmap_utils.py')
    reader=importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
    rows=[]
    for sid in scene_ids:
        p=Path(root)/sid/'dslr'
        images={im.name:im for im in reader.read_images_text(p/'colmap/images.txt').values()}
        frames=json.loads((p/'nerfstudio/transforms_undistorted.json').read_text())['frames']
        errors=[]
        for frame in frames:
            im=images[frame['file_path']]
            w2c=np.eye(4); w2c[:3,:3]=im.qvec2rotmat(); w2c[:3,3]=im.tvec
            errors.append(float(np.max(np.abs(scannetpp_pose_in_mesh_frame(frame['transform_matrix'])-np.linalg.inv(w2c)))))
        if not errors or max(errors)>1e-8:
            raise ValueError(f'pose verification failed: {sid}')
        rows.append(dict(scene=sid,n_train_poses=len(errors),max_abs_matrix_error=max(errors)))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',default=str(ROOT/'data_download/scannetpp/data'))
    p.add_argument('--scenes',nargs='+',default=['7b04052ad0','3cb9f85891'])
    args=p.parse_args();print(json.dumps(validate(args.data_root,args.scenes),indent=2))
