

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['grid', 'chicagofswild'], required=True)
    parser.add_argument('--splits', nargs='+', type=int, choices=[1, 2, 3, 4], default=[1])
    parser.add_argument('--variants', nargs='+', choices=['full', 'base', 'no_taw', 'no_cs', 'no_meta', 'global_only', 'local_only', 'resnet18'], default=['full', 'base'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 17, 27])
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output-root', default='outputs/experiments')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.dataset == 'grid' and 'resnet18' in args.variants:
        parser.error('resnet18 is a ChicagoFSWild variant')
    changes = {
        'full': [], 'base': ['train.mode=base', 'train.warmup_epochs=0', 'train.epochs=30', 'meta.use_taw=false', 'meta.use_global=false', 'meta.use_local=false'],
        'no_taw': ['meta.use_taw=false'], 'no_cs': ['meta.use_global=false', 'meta.use_local=false'],
        'no_meta': ['train.mode=no_meta'], 'global_only': ['meta.use_local=false'],
        'local_only': ['meta.use_global=false'], 'resnet18': ['model.feature_dim=512'],
    }
    splits = args.splits if args.dataset == 'grid' else [None]
    for split in splits:
        for variant in args.variants:
            partition = f'split{split}' if split else ('resnet18' if variant == 'resnet18' else 'resnet50')
            manifest = Path(args.data_root) / args.dataset / partition / 'features.jsonl'
            for seed in args.seeds:
                output = Path(args.output_root) / args.dataset / partition / variant / f'seed{seed}'
                command = [sys.executable, 'scripts/train.py', '--config', f'configs/{args.dataset}.yaml']
                for override in [f'seed={seed}', f'device={args.device}', f'data.manifest={manifest}', f'output_dir={output}', *changes[variant]]:
                    command += ['--set', override]
                commands = [command]
                if args.evaluate_test:
                    commands.append([sys.executable, 'scripts/evaluate.py', '--checkpoint', str(output / 'best.pt'), '--split', 'test', '--device', args.device, '--output', str(output / 'test.json')])
                for command in commands:
                    print(shlex.join(command), flush=True)
                    if args.execute:
                        subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
