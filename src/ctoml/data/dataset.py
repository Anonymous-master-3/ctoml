import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def validate_splits(records):

    seen, identities = set(), {}
    for row in records:
        sid = row.get('sample_id')
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError(f'invalid or duplicate sample_id: {sid!r}')
        seen.add(sid)
        split = row.get('split')
        if split not in ('train', 'dev', 'test'):
            raise ValueError(f'{sid}: split must be train, dev or test')
        performer = row.get('performer_id')
        if not isinstance(performer, str) or not performer:
            raise ValueError(f'{sid}: performer_id must be a nonempty string')
        if performer in identities and identities[performer] != split:
            raise ValueError(f'{sid}: performer {performer} leaks across splits')
        identities[performer] = split
        if not isinstance(row.get('transcript'), str):
            raise ValueError(f'{sid}: transcript must be a string')
        domain = row.get('domain_id', performer)
        if not isinstance(domain, (str, int)) or isinstance(domain, bool) or domain == '':
            raise ValueError(f'{sid}: domain_id must be a nonempty string or integer')
    return True


def load_records(path):
    path = Path(path).resolve()
    records = []
    with path.open(encoding='utf-8') as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('each row must be an object')
                for key in ('feature_path', 'feature_path_flipped', 'video_path'):
                    if key in row:
                        row[key] = str((path.parent / row[key]).resolve())
                row.setdefault('domain_id', row.get('performer_id'))
                records.append(row)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{path}:{lineno}: {exc}') from exc
    if not records:
        raise ValueError(f'{path}: empty manifest')
    validate_splits(records)
    return records


class FeatureDataset(Dataset):
    def __init__(self, records, vocab, augment=False):
        self.records = list(records)
        validate_splits(self.records)
        self.vocab = vocab
        self.augment = augment

        keys = {(type(r.get('domain_id', r['performer_id'])).__name__, str(r.get('domain_id', r['performer_id']))) for r in self.records}
        self.domain_to_index = {key: i for i, key in enumerate(sorted(keys))}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        sid = row['sample_id']
        try:
            path_key = 'feature_path'
            if self.augment and row['split'] == 'train' and 'feature_path_flipped' in row and torch.rand(()).item() < .5:
                path_key = 'feature_path_flipped'
            path = Path(row[path_key])
            if path.suffix == '.npy':
                array = np.load(path, allow_pickle=False)
                features = torch.from_numpy(np.array(array, copy=True))
            elif path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as archive:
                    if 'features' not in archive:
                        raise ValueError('npz requires features key')
                    features = torch.from_numpy(np.array(archive['features'], copy=True))
            elif path.suffix == '.pt':
                features = torch.load(path, map_location='cpu', weights_only=True)
                if isinstance(features, dict):
                    features = features['features']
            else:
                raise ValueError('feature suffix must be .npy, .npz or .pt')
            if not isinstance(features, torch.Tensor) or features.ndim != 2 or min(features.shape) < 1:
                raise ValueError('features must have nonempty shape [T,F]')
            if not features.is_floating_point() or not torch.isfinite(features).all():
                raise ValueError('features must be finite floating point values')
            features = features.float()
            if not torch.isfinite(features).all():
                raise ValueError('features exceed finite float32 range')
            for key, expected in [('num_frames', features.shape[0]), ('feature_dim', features.shape[1])]:
                if not isinstance(row.get(key), int) or isinstance(row[key], bool) or row[key] != expected:
                    raise ValueError(f'{key} must equal actual feature shape ({expected})')
            extractor = row.get('extractor')
            if not isinstance(extractor, dict) or not extractor.get('name'):
                raise ValueError('extractor metadata with name is required')
        except (KeyError, ValueError, TypeError, OSError, RuntimeError) as exc:
            raise ValueError(f'{sid}: invalid features: {exc}') from exc
        domain = row.get('domain_id', row['performer_id'])
        return {'features': features.float(), 'tokens': self.vocab.encode(row['transcript']),
                'domain': self.domain_to_index[(type(domain).__name__, str(domain))],
                'sample_id': sid, 'transcript': row['transcript'], 'performer_id': row['performer_id']}


ManifestDataset = FeatureDataset


def collate_samples(samples):
    if not samples:
        raise ValueError('cannot collate an empty batch')
    dimensions = {s['features'].shape[1] for s in samples}
    if len(dimensions) != 1:
        raise ValueError('batch has inconsistent feature dimensions')
    b, t, f = len(samples), max(s['features'].shape[0] for s in samples), dimensions.pop()
    length = max(len(s['tokens']) + 1 for s in samples)
    features = torch.zeros(b, t, f)
    targets, inputs = torch.zeros(b, length, dtype=torch.long), torch.zeros(b, length, dtype=torch.long)
    for i, sample in enumerate(samples):
        features[i, :len(sample['features'])] = sample['features']
        tokens = sample['tokens']
        targets[i, :len(tokens) + 1] = torch.tensor(tokens + [2])
        inputs[i, :len(tokens) + 1] = torch.tensor([1] + tokens)
    return {'features': features, 'feature_lengths': torch.tensor([len(s['features']) for s in samples]),
            'decoder_inputs': inputs, 'targets': targets,
            'target_lengths': torch.tensor([len(s['tokens']) + 1 for s in samples]),
            'domains': torch.tensor([s['domain'] for s in samples]),
            'sample_ids': [s['sample_id'] for s in samples], 'transcripts': [s['transcript'] for s in samples],
            'performer_ids': [s['performer_id'] for s in samples]}


def sample_episode(dataset, batch_size, rng: random.Random, test_domain_count=1, partition_seed=None):





    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    groups = defaultdict(list)
    for index, row in enumerate(dataset.records):
        if row['split'] == 'train':
            domain = row.get('domain_id', row['performer_id'])
            groups[(type(domain).__name__, str(domain))].append(index)
    domains = sorted(groups)
    if not 1 <= test_domain_count < len(domains):
        raise ValueError('episode requires nonempty disjoint meta-train/meta-test source domains')
    (random.Random(partition_seed) if partition_seed is not None else rng).shuffle(domains)
    test_domains, train_domains = domains[:test_domain_count], domains[test_domain_count:]

    def draw(partition):
        chosen = []
        while len(chosen) < batch_size:
            order = list(partition)
            rng.shuffle(order)
            for domain in order:
                chosen.append(rng.choice(groups[domain]))
                if len(chosen) == batch_size:
                    break
        return collate_samples([dataset[i] for i in chosen])
    return draw(train_domains), draw(test_domains)
