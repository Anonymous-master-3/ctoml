import json
import random

import numpy as np
import pytest
import torch

from ctoml.data import FeatureDataset, Vocabulary, collate_samples, load_records, sample_episode, validate_splits
from tests._data import build_dataset


def test_vocabulary_controls_unknown_and_character_roundtrip():
    vocab = Vocabulary.from_texts(['red blue', 'red'])
    assert vocab.encode('blue red') == [4, 5]
    assert vocab.encode('unseen') == [3]
    assert vocab.decode([1, 4, 5, 2, 4]) == 'blue red'
    assert Vocabulary.from_dict(vocab.to_dict()).to_dict() == vocab.to_dict()
    chars = Vocabulary.from_texts(['a- b'], mode='character')
    assert chars.decode(chars.encode('a- b')) == 'a- b'


def test_shapes_learning_signal_and_episode_isolation(tmp_path):
    records = load_records(build_dataset(tmp_path))
    vocab = Vocabulary.from_texts(r['transcript'] for r in records if r['split'] == 'train')
    dataset = FeatureDataset(records, vocab)
    batch = collate_samples([dataset[0], dataset[1], dataset[2]])
    assert batch['feature_lengths'].unique().numel() > 1
    assert batch['targets'][0, 1].item() == 2
    assert batch['targets'][0, 2].item() == 0
    assert batch['decoder_inputs'][:, 0].eq(1).all()

    word_index = {'blue': 0, 'green': 1, 'red': 2, 'white': 3}
    for row, sample in zip(records, dataset):
        visual = sample['features'][:, :4].argmax(-1).tolist()
        assert visual[0] == word_index[row['transcript'].split()[0]]
    train, test = sample_episode(dataset, 8, random.Random(5))
    assert set(train['domains'].tolist()).isdisjoint(test['domains'].tolist())
    source_ids = {r['sample_id'] for r in records if r['split'] == 'train'}
    assert set(train['sample_ids'] + test['sample_ids']) <= source_ids
    train2, test2 = sample_episode(dataset, 8, random.Random(5))
    assert train['sample_ids'] == train2['sample_ids']
    assert test['sample_ids'] == test2['sample_ids']


def test_identity_leakage_and_duplicate_rejected(tmp_path):
    records = load_records(build_dataset(tmp_path))
    with pytest.raises(ValueError, match='duplicate'):
        validate_splits(records + [records[0]])
    other = dict(records[-1], performer_id=records[0]['performer_id'])
    with pytest.raises(ValueError, match='leaks'):
        validate_splits([records[0], other])


@pytest.mark.parametrize('extension', ['npy', 'npz', 'pt'])
def test_safe_formats_and_metadata_validation(tmp_path, extension):
    records = load_records(build_dataset(tmp_path))
    row = dict(records[0])
    array = np.load(row['feature_path'], allow_pickle=False)
    path = tmp_path / ('sample.' + extension)
    if extension == 'npy':
        np.save(path, array)
    elif extension == 'npz':
        np.savez(path, features=array)
    else:
        torch.save({'features': torch.tensor(array)}, path)
    row['feature_path'] = str(path)
    vocab = Vocabulary.from_texts([row['transcript']])
    assert torch.equal(FeatureDataset([row], vocab)[0]['features'], torch.tensor(array))
    with pytest.raises(ValueError, match='num_frames'):
        FeatureDataset([dict(row, num_frames=999)], vocab)[0]
    row['extractor'] = {}
    with pytest.raises(ValueError, match='extractor'):
        FeatureDataset([row], vocab)[0]


def test_nan_and_pickle_arrays_rejected(tmp_path):
    records = load_records(build_dataset(tmp_path))
    row = records[0]
    vocab = Vocabulary.from_texts([row['transcript']])
    np.save(row['feature_path'], np.full((row['num_frames'], row['feature_dim']), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match='finite'):
        FeatureDataset([row], vocab)[0]
    np.save(row['feature_path'], np.array([object()], dtype=object))
    with pytest.raises(ValueError, match='allow_pickle'):
        FeatureDataset([row], vocab)[0]


def test_empty_transcript_has_eos_and_single_domain_episode_fails(tmp_path):
    records = load_records(build_dataset(tmp_path))
    row = dict(records[0], transcript='')
    dataset = FeatureDataset([row], Vocabulary.from_texts(['red']))
    batch = collate_samples([dataset[0]])
    assert batch['targets'].tolist() == [[2]]
    assert batch['decoder_inputs'].tolist() == [[1]]
    with pytest.raises(ValueError, match='disjoint'):
        sample_episode(dataset, 2, random.Random(1))


def test_flip_augmentation_only_on_train(tmp_path, monkeypatch):
    records = load_records(build_dataset(tmp_path))
    train_row, test_row = dict(records[0]), dict(records[-1])
    for row in [train_row, test_row]:
        path = tmp_path / (row['sample_id'] + '_flip.npy')
        np.save(path, np.full((row['num_frames'], row['feature_dim']), 42, dtype=np.float32))
        row['feature_path_flipped'] = str(path)
    vocab = Vocabulary.from_texts([train_row['transcript']])
    monkeypatch.setattr(torch, 'rand', lambda *args: torch.tensor(.1))
    augmented = FeatureDataset([train_row, test_row], vocab, augment=True)
    assert augmented[0]['features'].eq(42).all()
    assert not augmented[1]['features'].eq(42).any()
    assert not FeatureDataset([train_row], vocab)[0]['features'].eq(42).any()
