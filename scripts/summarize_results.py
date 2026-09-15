

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.root).rglob('test.json')):
        result = json.loads(path.read_text())
        metrics = result['metrics']
        rows.append({'run': str(path.parent), 'checkpoint': result['checkpoint'], 'epoch': result['epoch'],
                     'split': result['split'], 'samples': metrics['samples'],
                     **{name + '_percent': None if metrics[name] is None else 100 * metrics[name] for name in ['wer', 'cer', 'letter_accuracy']}})
    if not rows:
        parser.error('No actual test.json evaluation outputs found')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} measured runs to {output}')


if __name__ == '__main__':
    main()
