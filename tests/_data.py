import json
from pathlib import Path

import numpy as np


def build_dataset(output_dir, seed=7, feature_dim=16, samples_per_domain=12):
    if feature_dim < 4 or samples_per_domain < 1:
        raise ValueError('feature_dim >= 4 and samples_per_domain >= 1 are required')
    root = Path(output_dir).resolve()
    (root / 'features').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    words = ['blue', 'green', 'red', 'white']
    centers = np.zeros((4, feature_dim), dtype=np.float32)
    centers[:, :4] = np.eye(4, dtype=np.float32) * 3
    records = []
    for domain, split in enumerate(['train', 'train', 'train', 'train', 'dev', 'test']):
        performer = f'domain_{domain}'
        offset = rng.normal(0, .025, feature_dim)
        available = list(range(4)) if domain != 1 else [0, 1, 2]
        for index in range(samples_per_domain):
            size = index % 3 + 1
            ids = [available[(index + position) % len(available)] for position in range(size)]
            if index % 4 == 0:
                ids = [available[index % len(available)]] * size
            feature = np.concatenate([
                np.repeat(centers[token][None], 2 + ((index + token) % 2), axis=0)
                for token in ids
            ])
            feature = (feature + offset + rng.normal(0, .015, feature.shape)).astype(np.float32)
            sample_id = f'{performer}_{index:04d}'
            relative_path = f'features/{sample_id}.npy'
            np.save(root / relative_path, feature, allow_pickle=False)
            records.append({
                'sample_id': sample_id,
                'performer_id': performer,
                'domain_id': performer,
                'split': split,
                'transcript': ' '.join(words[token] for token in ids),
                'feature_path': relative_path,
                'num_frames': len(feature),
                'feature_dim': feature_dim,
                'extractor': {
                    'name': 'test-centers',
                    'weights_sha256': None,
                    'layer': 'centers',
                    'preprocessing': {'seed': seed},
                },
            })
    path = root / 'manifest.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
    return path
