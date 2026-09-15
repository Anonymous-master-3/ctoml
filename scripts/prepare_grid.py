
import argparse
from ctoml.features.manifests import grid_records, write_manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video-root', required=True)
    p.add_argument('--align-root', required=True)
    p.add_argument('--split-id', type=int, choices=range(1, 5), required=True)
    p.add_argument('--dev-speakers', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    rows = grid_records(a.video_root, a.align_root, a.split_id, a.dev_speakers)
    write_manifest(rows, a.output)
    print('Wrote', len(rows), 'records to', a.output)
