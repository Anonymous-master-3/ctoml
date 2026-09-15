
import argparse
from ctoml.features.manifests import chicago_records, write_manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--frame-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    rows = chicago_records(a.csv, a.frame_root)
    write_manifest(rows, a.output)
    print('Wrote', len(rows), 'records to', a.output)
