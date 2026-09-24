"""Archive small reproducibility evidence, excluding datasets, weights and checkpoints."""
import difflib,hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[2];base=root/'experiments/out/EXP0022_sgs_sunpatch';dest=root/'research/evidence/EXP0024'
# Renumbered 2026-09-23: EXP0022/EXP0023 were independently assigned the same
# night to the Cali-HDR/Pano2Pano dataset audit (already committed); this SGS
# sun-patch pairing run (working dir name kept as-is, gitignored) archives to
# EXP0024 instead to avoid overwriting that evidence. See DECISIONS.md D0013.
assert json.loads((base/'pair_completed.json').read_text())['completed']
dest.mkdir(parents=True,exist_ok=False)
for name in ['input_hashes.json','rgbx_weights.json','rebuild_check_v2.json','PORTABILITY.json','color_contract.json','prior_only_diagnostic.json','pair_completed.json','pair_v2.log']:
    shutil.copy2(base/name,dest/name)
for name in ['input.json','prior_manifest.json']:
    shutil.copy2(base/'input_v2'/name,dest/name)
for condition in ['sun_a','sky']:
    name=f'{condition}_v2_1000_600';d=dest/name;d.mkdir()
    for f in ['config.json','result.json','audit.json','history.jsonl']:shutil.copy2(base/name/f,d/f)
    shutil.copy2(base/(name+'.log'),d/'training.log')
shutil.copytree(base/'evaluation_v2',dest/'evaluation')
invalid=dest/'excluded_initial_run';invalid.mkdir()
for f in ['INVALID_SCIENTIFIC_RESULT.json','config.json','result.json']:
    shutil.copy2(base/'sun_a_1000_600'/f,invalid/f)
sources=[]
for name in ['prepare_sgs_compat.py','prepare_sgs_sunpatch.py','infer_sgs_priors.py','run_sgs_p1_pilot.py','launch_sgs_p1.py','evaluate_sgs_sunpatch.py','audit_sgs_p1_run.py','run_sgs_sunpatch_pair.py','report_sgs_sunpatch.py','test_sgs_color_contract.py','audit_sgs_priors.py','archive_sgs_sunpatch.py']:
    p=root/'experiments/baselines'/name;sources.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(dest/'script_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
patch=[]
for row in json.loads((base/'rebuild_check_v2.json').read_text()):
    f=row['file'];old=root/'third_party/sgs_intrinsic'/f;new=root/'third_party/sgs_adapted'/f
    before=old.read_text().splitlines(keepends=True) if old.is_file() else []
    patch.extend(difflib.unified_diff(before,new.read_text().splitlines(keepends=True),fromfile='official/'+f,tofile='adapted/'+f))
(dest/'sgs_portability.patch').write_text(''.join(patch))
(dest/'README.md').write_text('''# EXP0024 evidence

SGS public-upstream P1 material-stage adaptation, not a full official reproduction.
The accepted pair is `sun_a_v2_1000_600` and `sky_v2_1000_600`.
`excluded_initial_run` omitted required RGB-X input linearization and is not scientific evidence.
Training never reads material/solar ground truth; the independent evaluation does.
Weights, input images/mesh, and large checkpoints remain local; hashes are recorded.
Code lives in `experiments/baselines/`; source revisions are pinned in its lock file.
Full Chinese interpretation: `research/SGS_SUNPATCH_BENCHMARK_ZH.md`.

Scene attribution: Country Kitchen by Jay-Artist, CC BY 3.0, curated by Benedikt Bitterli,
https://noobody.org/resources/ . Floor material and lighting were modified for this experiment.
''')
print(dest)
