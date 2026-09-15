




import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ctoml.features.extractors import ResNetExtractor, AVHubertExtractor, sha256_file
from ctoml.features.pipeline import extract_manifest
from ctoml.features.preprocess import DlibMouthCropper


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--extractor', choices=['resnet18','resnet50','avhubert'], required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--avhubert-repo')
    p.add_argument('--layer', type=int)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--roi', choices=['none','mouth','face'], required=True)
    p.add_argument('--landmark-predictor')
    p.add_argument('--face-boxes')
    p.add_argument('--face-mat-root')
    p.add_argument('--cache-flips', action='store_true')
    a = p.parse_args()
    if a.extractor == 'avhubert':
        if not a.avhubert_repo or a.pretrained:
            p.error('AV-HuBERT needs --avhubert-repo and --checkpoint; --pretrained is ResNet-only')
        extractor = AVHubertExtractor(a.avhubert_repo,a.checkpoint,a.layer,a.device)
    else:
        extractor = ResNetExtractor(a.extractor,a.checkpoint,a.pretrained,a.device,a.batch_size)
    cropper, predictor_hash = None,None
    if a.roi == 'mouth':
        if not a.landmark_predictor:
            p.error('--roi mouth needs --landmark-predictor')
        cropper = DlibMouthCropper(a.landmark_predictor)
        predictor_hash = sha256_file(a.landmark_predictor)
    boxes = json.loads(Path(a.face_boxes).read_text()) if a.face_boxes else {}
    if a.face_mat_root:
        from scipy.io import loadmat
        for line in Path(a.manifest).read_text().splitlines():
            if line.strip():
                sid = json.loads(line)['sample_id']
                boxes[sid] = loadmat(Path(a.face_mat_root)/sid/'face.mat')['face'].reshape(-1).tolist()
    rows = extract_manifest(a.manifest,a.output,a.cache_dir,extractor,roi=a.roi,
                            mouth_cropper=cropper,face_boxes=boxes,cache_flips=a.cache_flips,
                            predictor_sha256=predictor_hash)
    print('Wrote', len(rows), 'feature records to', a.output)

if __name__ == '__main__':
    main()
