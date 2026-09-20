"""Record acquired PBR assets and unresolved daylight conversion requirements."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(name):
    base = ROOT / 'data_download/pbr_interiors'
    archive = base / f'{name}.zip'
    folder = base / name / name
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError(f'CRC failure: {bad}')
    source = folder / 'scene_v3.xml'
    tree = ET.parse(source)
    files = sorted({e.attrib['value'] for e in tree.iter('string')
                    if e.get('name') == 'filename'})
    missing = [p for p in files if not (folder / p).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return dict(scene=name, provenance='artist_authored_not_real_scan',
                source_url=f'https://noobody.org/resources/mitsuba/{name}.zip',
                archive_sha256=sha256(archive), zip_crc_pass=True,
                license_text=(folder / 'LICENSE.txt').read_text(),
                source_xml_sha256=sha256(source),
                referenced_files={p: sha256(folder / p) for p in files},
                shape_count=len(list(tree.iter('shape'))),
                bsdf_types=sorted({e.get('type') for e in tree.iter('bsdf')}),
                emitter_types=[e.get('type') for e in tree.iter('emitter')],
                solar_dataset_ready=False,
                pending=['inspect actual window geometry and transmission',
                         'replace proxy lighting with explicit recorded sun and sky',
                         'render and inspect convergence and appearance',
                         'export photo-only inputs separately from ground truth'])


if __name__ == '__main__':
    out = ROOT / 'research/evidence/EXP0013_preparation'
    out.mkdir(parents=True, exist_ok=True)
    records = [audit(name) for name in ('bathroom', 'kitchen')]
    (out / 'asset_audit.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps([{k: r[k] for k in ('scene', 'shape_count', 'emitter_types',
                                        'solar_dataset_ready')} for r in records]))
