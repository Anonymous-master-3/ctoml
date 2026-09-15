
import csv
import json
import re
from pathlib import Path

GRID_TEST_SPEAKERS = ({'s1','s2','s20','s22'}, {'s3','s4','s23','s24'},
                      {'s5','s6','s25','s26'}, {'s7','s8','s27','s28'})


def write_manifest(records, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ids = [r['sample_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate sample_id')
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))


def grid_records(video_root, align_root, split_id, dev_speakers):
    if split_id not in range(1, 5):
        raise ValueError('GRID split_id must be 1..4')
    test = GRID_TEST_SPEAKERS[split_id - 1]
    dev = set(dev_speakers)
    if not dev or dev & test:
        raise ValueError('Specify nonempty dev speakers disjoint from held-out test speakers')
    video_root, align_root = Path(video_root), Path(align_root)
    def speaker(path, root):
        matches = [p for p in path.relative_to(root).parts if re.fullmatch(r's\d+', p)]
        if len(matches) != 1:
            raise ValueError('Expected exactly one sN directory in ' + str(path))
        return matches[0]
    videos = {}
    for p in sorted(video_root.rglob('*')):
        if p.suffix.lower() not in {'.mpg', '.mpeg', '.mp4', '.avi', '.mov'}:
            continue
        key = (speaker(p, video_root), p.stem)
        if key in videos:
            raise ValueError('Ambiguous video: ' + str(key))
        videos[key] = p
    records = []
    for p in sorted(align_root.rglob('*.align')):
        person = speaker(p, align_root)
        video = videos.get((person, p.stem))
        if video is None:
            raise FileNotFoundError('Missing video for ' + str(p))
        words = []
        previous_end = -1
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            start, end, word = line.split()
            start, end = int(start), int(end)
            if start < previous_end or end <= start:
                raise ValueError('Invalid alignment intervals: ' + str(p))
            previous_end = end
            if word.lower() not in {'sil', 'sp'}:
                words.append(word.lower())
        if not words:
            raise ValueError('Empty lexical alignment: ' + str(p))
        records.append(dict(sample_id=person + '/' + p.stem, performer_id=person,
                            domain_id=person, split='test' if person in test else 'dev' if person in dev else 'train',
                            transcript=' '.join(words), video_path=str(video.resolve()),
                            dataset='GRID', protocol='grid-' + str(split_id), alignment_path=str(p.resolve())))
    people = {r['performer_id'] for r in records}
    if not test <= people or not dev <= people or not people - test - dev:
        raise ValueError('Missing required test/dev speakers or no training speakers')
    return records


def chicago_records(csv_path, frame_root):
    root = Path(frame_root).resolve()
    records, identities = [], {}
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'filename','number_of_frames','label_proc','partition','signer'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Missing official CSV fields: ' + str(required - set(reader.fieldnames or [])))
        for row in reader:
            sid, person, split = row['filename'].strip(), row['signer'].strip(), row['partition'].strip()
            n = int(row['number_of_frames'])
            if split not in {'train','dev','test'} or not sid or not person or n < 1 or not row['label_proc']:
                raise ValueError('Invalid Chicago metadata row: ' + str(row))
            if person in identities and identities[person] != split:
                raise ValueError('Signer leakage across official partitions: ' + person)
            identities[person] = split
            path = (root / sid).resolve()
            if root not in path.parents or not path.is_dir():
                raise FileNotFoundError('Expected frame directory under frame_root: ' + str(path))
            records.append(dict(sample_id=sid, performer_id=person, domain_id=person, split=split,
                                transcript=row['label_proc'], num_frames=n, video_path=str(path),
                                dataset='ChicagoFSWild', protocol='official'))
    if not records:
        raise ValueError('Empty Chicago CSV')
    return records
