from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from ctoml.config import DEFAULTS, load_config, validate
from ctoml.data import load_records
from ctoml.training.checkpoint import load_checkpoint
from ctoml.training.engine import evaluate_checkpoint, train
from tests._data import build_dataset


def tiny_config(tmp_path, mode='meta', samples_per_domain=3):
    manifest = build_dataset(tmp_path / 'data', feature_dim=4, samples_per_domain=samples_per_domain)
    config = deepcopy(DEFAULTS)
    config['output_dir'] = str(tmp_path / 'run')
    config['data'].update(manifest=str(manifest), horizontal_flip=False)
    config['model'].update(feature_dim=4, d_model=8, nhead=2, dim_feedforward=16, dropout=.1, max_length=16)
    config['train'].update(mode=mode, warmup_epochs=1, epochs=2, batch_size=2, steps_per_epoch=1,
                           warmup_lr=.01, outer_lr=.003)
    config['meta'].update(inner_steps=2, inner_lr=.001, max_pairs=4)
    config['decode']['max_length'] = 5
    return validate(config)


def assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_identical(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('stop_epoch', [1, 2])
def test_epoch_boundary_resume_matches_continuous_exactly(tmp_path, stop_epoch):
    config = tiny_config(tmp_path)
    continuous = deepcopy(config)
    continuous['output_dir'] = str(tmp_path / 'continuous')
    train(continuous)
    expected = load_checkpoint(Path(continuous['output_dir']) / 'last.pt')
    train(config, stop_after_epochs=stop_epoch)
    checkpoint = Path(config['output_dir']) / 'last.pt'
    intermediate = load_checkpoint(checkpoint)
    assert intermediate['epoch_completed'] == stop_epoch
    train(config, resume=checkpoint)
    actual = load_checkpoint(checkpoint)
    for key in ['model', 'optimizer', 'weights', 'weights_ready', 'history', 'rng', 'step', 'best_metric']:
        assert_identical(actual[key], expected[key])
    assert actual['epoch_completed'] == 3
    best = load_checkpoint(Path(config['output_dir']) / 'best.pt')
    assert best['history'][-1]['stage'] == 'meta'


@pytest.mark.parametrize('mode', ['base', 'no_meta', 'no_cs'])
def test_training_modes_run_and_disable_expected_losses(tmp_path, mode):
    config = tiny_config(tmp_path, mode='meta' if mode == 'no_cs' else mode)
    config['train'].update(warmup_epochs=0, epochs=1)
    if mode == 'no_cs':
        config['meta'].update(use_global=False, use_local=False)
    result = train(config)
    entry = result['history'][-1]
    assert torch.isfinite(torch.tensor(entry['train']['total']))
    state = load_checkpoint(Path(config['output_dir']) / 'last.pt')
    if mode in ('base', 'no_cs'):
        assert entry['train']['global'] == 0
        assert entry['train']['local'] == 0
    assert state['weights_ready'] is (mode != 'base')
    assert (Path(config['output_dir']) / 'best.pt').is_file()


def test_learnable_visual_sequences_reduce_loss_and_error(tmp_path):
    config = tiny_config(tmp_path, mode='base', samples_per_domain=4)
    config['model'].update(d_model=16, nhead=2, dim_feedforward=32, dropout=0.)
    config['train'].update(epochs=20, batch_size=4, steps_per_epoch=4, warmup_lr=.02)
    result = train(config)
    initial, final = result['history'][0]['dev'], result['history'][-1]['dev']
    assert final['loss'] < initial['loss'] * .7
    assert final['wer'] < initial['wer']
    evaluation = evaluate_checkpoint(Path(config['output_dir']) / 'best.pt')
    assert evaluation['metrics']['samples'] == 4
    assert len(evaluation['predictions']) == 4


def test_replacement_manifest_cannot_disguise_seen_performers(tmp_path):
    config = tiny_config(tmp_path, mode='base')
    config['train']['epochs'] = 1
    train(config)
    records = load_records(config['data']['manifest'])
    forged = dict(next(row for row in records if row['split'] == 'train'), split='test')
    manifest = tmp_path / 'forged.jsonl'
    manifest.write_text(json.dumps(forged) + '\n')
    with pytest.raises(ValueError, match='present in training'):
        evaluate_checkpoint(Path(config['output_dir']) / 'best.pt', manifest=manifest)


def test_resume_rejects_changed_inputs_and_hyperparameters(tmp_path):
    config = tiny_config(tmp_path, mode='base')
    train(config, stop_after_epochs=1)
    checkpoint = Path(config['output_dir']) / 'last.pt'
    changed = deepcopy(config)
    changed['train']['warmup_lr'] *= 2
    with pytest.raises(ValueError, match='Resume config changed'):
        train(changed, resume=checkpoint)
    with Path(config['data']['manifest']).open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='changed since checkpoint'):
        train(config, resume=checkpoint)


@pytest.mark.parametrize('section,key,value', [
    ('train', 'mode', 'unknown'), ('train', 'epochs', 0),
    ('meta', 'inner_steps', 0), ('meta', 'temperature', 0),
    ('model', 'd_model', 7), ('data', 'horizontal_flip', 'yes'),
])
def test_invalid_configuration_is_rejected(section, key, value):
    config = deepcopy(DEFAULTS)
    config[section][key] = value
    with pytest.raises(ValueError):
        validate(config)


def test_config_unknown_keys_and_inheritance_cycles(tmp_path):
    path = tmp_path / 'invalid.yaml'
    path.write_text('train:\n  bath_size: 2\n')
    with pytest.raises(ValueError, match='Unknown configuration key'):
        load_config(path)
    path.write_text('extends: invalid.yaml\n')
    with pytest.raises(ValueError, match='cycle'):
        load_config(path)
