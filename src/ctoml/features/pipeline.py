
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .extractors import sha256_file
from .preprocess import read_frames, face_roi
from .manifests import write_manifest


def source_digest(path):
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if not files:
        raise ValueError('No image frames in ' + str(path))
    for p in files:
        digest.update(p.name.encode())
        digest.update(sha256_file(p).encode())
    return digest.hexdigest()


def extract_manifest(manifest, output, cache_dir, extractor, roi='none', mouth_cropper=None,
                     face_boxes=None, cache_flips=False, predictor_sha256=None):
    manifest = Path(manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty input manifest')
    if roi not in {'none','mouth','face'}:
        raise ValueError('Unknown ROI policy')
    if roi == 'mouth' and mouth_cropper is None:
        raise ValueError('mouth ROI requires a landmark predictor')
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        source = Path(row['video_path'])
        source = source if source.is_absolute() else manifest.parent / source
        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        box = (face_boxes or {}).get(row['sample_id'])
        if roi == 'face' and box is None:
            raise ValueError('No sequence-level signer face box for ' + row['sample_id'])
        preprocessing = dict(extractor.metadata['preprocessing'], roi=roi, face_box=box,
                             landmark_predictor_sha256=predictor_sha256,
                             implementation='ctoml-roi-v1')
        metadata = dict(extractor.metadata, preprocessing=preprocessing)
        provenance = dict(path=str(source), sha256=source_digest(source))
        frames = list(read_frames(source))
        if not frames:
            raise ValueError('No decoded frames for ' + row['sample_id'])


        source_count = row.get('source_num_frames')
        if source_count is None and 'feature_path' not in row:
            source_count = row.get('num_frames')
        if source_count is not None and source_count != len(frames):
            raise ValueError('Official/source frame count mismatch for ' + row['sample_id'])
        if roi == 'mouth':
            frames = [mouth_cropper(frame) for frame in frames]
        elif roi == 'face':
            frames = [face_roi(frame, box) for frame in frames]
        result = dict(row, video_path=str(source), source=provenance, extractor=metadata,
                      source_num_frames=len(frames))

        for old_key in ('feature_path', 'feature_path_flipped', 'num_frames', 'feature_dim'):
            result.pop(old_key, None)
        variants = [False,True] if cache_flips and row['split'] == 'train' else [False]
        for flip in variants:
            key = dict(sample_id=row['sample_id'], split=row['split'], protocol=row.get('protocol'),
                       source=provenance, extractor=metadata, flip=flip)
            digest = hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
            path = cache_dir / (digest + '.npy')

            if path.exists():
                array = np.load(path, allow_pickle=False)
            else:
                array = np.asarray(extractor([f[:,::-1].copy() for f in frames] if flip else frames), dtype=np.float32)
                if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
                    raise ValueError('Extractor must return finite nonempty [T,F] features')
                temporary = path.with_suffix('.tmp')
                with open(temporary, 'wb') as f:
                    np.save(f,array,allow_pickle=False)
                os.replace(temporary,path)
            if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
                raise ValueError('Corrupt cached features: ' + str(path))
            if flip and tuple(array.shape) != (result['num_frames'],result['feature_dim']):
                raise ValueError('Flipped feature shape differs')
            result['feature_path_flipped' if flip else 'feature_path'] = str(path)
            result.update(num_frames=int(array.shape[0]), feature_dim=int(array.shape[1]))
        results.append(result)
    write_manifest(results, output)
    return results
