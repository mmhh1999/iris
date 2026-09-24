"""Prior-only training-view diagnostic; separate from heldout SGS scores."""
import json
from pathlib import Path
import cv2,numpy as np
root=Path(__file__).resolve().parents[2];base=root/'experiments/out/EXP0022_sgs_sunpatch';data=base/'input_v2'
manifest=json.loads((data/'input.json').read_text());rows=[]
for condition in ['sun_a','sky']:
    for frame in manifest['frames']:
        if frame['split']!='train':continue
        view=frame['view'];truth=np.load(root/f'experiments/out/EXP0019_sunpatch/data_03/generator_truth/sun_a_view{view:02d}.npz')
        # Public RGB-X applies gamma 1/2.2; invert exactly for this prior-only audit.
        pred=np.load(data/condition/'priors'/f'view{view:02d}.npz')['albedo'].transpose(1,2,0)**2.2
        floor=truth['floor'];kernel=np.ones((3,3),'uint8')
        lit=cv2.erode((floor&truth['sun_visible']).astype('uint8'),kernel).astype(bool)
        shadow=cv2.erode((floor&~truth['sun_visible']).astype('uint8'),kernel).astype(bool)
        if min(lit.sum(),shadow.sum())<32:continue
        gray=pred.mean(-1)
        rows.append(dict(condition=condition,view=view,lit_count=int(lit.sum()),shadow_count=int(shadow.sum()),linear_albedo_mae=float(np.abs(pred[floor]-[.28,.20,.12]).mean()),normalized_gap=float((gray[lit].mean()-gray[shadow].mean())/gray[floor].mean())))
report=dict(scope='Training-view prior-only diagnostic, not heldout benchmark or SGS result',rows=rows)
(base/'prior_only_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
