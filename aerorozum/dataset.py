"""Audit/normalize the supplied archive without altering original evidence."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from .common import config, data_root, digest, save, utcnow


def audit(archive=None, export=False):
    archive = Path(archive or config()['dataset_zip'])
    counts, sources, sizes = Counter(), Counter(), Counter()
    rows, errors = [], []
    with ZipFile(archive) as z:
        names = set(z.namelist())
        manifests = [n for n in names if n.endswith('/manifest.csv')]
        provenance = {}
        if manifests:
            for row in csv.DictReader(io.StringIO(z.read(manifests[0]).decode('utf-8-sig'))):
                provenance[row['filename']] = row
        for name in sorted(names):
            if not name.endswith('.json'):
                continue
            payload = json.loads(z.read(name))
            if not isinstance(payload, dict) or 'shapes' not in payload:
                continue
            image_name = str(PurePosixPath(name).parent / payload['imagePath'])
            if image_name not in names or '..' in PurePosixPath(image_name).parts:
                errors.append({'file': name, 'error': 'Missing or unsafe image reference'})
                continue
            width, height = payload['imageWidth'], payload['imageHeight']
            source = provenance.get(PurePosixPath(image_name).name, {}).get('annotation_source', 'unknown')
            zone = PurePosixPath(name).stem.split('__')[1]
            split = 'test' if zone in ('zone8', 'zone9') else 'validation' if zone == 'zone7' else 'train'
            boxes = []
            for shape in payload['shapes']:
                if shape.get('shape_type') != 'rectangle' or len(shape['points']) != 2:
                    errors.append({'file': name, 'error': 'Unsupported shape'})
                    continue
                (xa, ya), (xb, yb) = shape['points']
                counts['reordered_boxes'] += xa > xb or ya > yb
                x1, x2 = sorted((xa, xb)); y1, y2 = sorted((ya, yb))
                if x1 < -1e-9 or y1 < -1e-9 or x2 > width + 1e-9 or y2 > height + 1e-9 or x1 >= x2 or y1 >= y2:
                    errors.append({'file': name, 'error': 'Invalid box bounds'})
                    continue
                counts['epsilon_boundary_boxes'] += x1 < 0 or y1 < 0 or x2 > width or y2 > height
                box = dict(x_min=max(0, x1) * 1000 / width, y_min=max(0, y1) * 1000 / height, x_max=min(width, x2) * 1000 / width, y_max=min(height, y2) * 1000 / height)
                boxes.append(dict(label=shape['label'], bbox=box))
            raw = z.read(image_name)
            row = dict(id=PurePosixPath(name).stem, archive_image=image_name, width=width, height=height, zone=zone, split=split, provenance=source, boxes=boxes, sha256=hashlib.sha256(raw).hexdigest())
            rows.append(row)
            counts['images'] += 1; counts['boxes'] += len(boxes)
            counts['positive_images' if boxes else 'empty_images'] += 1
            counts[split + '_images'] += 1; sources[source] += 1; sizes[f'{width}x{height}'] += 1
            if export:
                target = data_root() / 'datasets/real' / split / (row['id'] + '.jpg')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                row['image'] = str(target.relative_to(data_root()))
    manifest = dict(schema_version=1, source=str(archive), created_at=utcnow(), status='passed' if not errors else 'failed', counts=dict(counts), provenance=dict(sources), image_sizes=dict(sizes), errors=errors, identity_verified=False, manifest_hash=digest(rows))
    save(data_root() / 'audits/dataset.json', manifest)
    save(data_root() / 'datasets/real/manifest.json', dict(metadata=manifest, images=rows))
    return manifest
