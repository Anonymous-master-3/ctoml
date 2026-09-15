# CtoML

## Installation

```bash
python3 -m pip install -e '.[vision]'
```

## Datasets

| Dataset | Download | Paper |
| --- | --- | --- |
| GRID | [Zenodo](https://zenodo.org/records/3625687) | [An Audio-Visual Corpus for Speech Perception and Automatic Speech Recognition](https://doi.org/10.1121/1.2229005) |
| ChicagoFSWild | [ChicagoFSWild.tgz](https://dl.ttic.edu/ChicagoFSWild.tgz) | [American Sign Language Fingerspelling Recognition in the Wild](https://doi.org/10.1109/SLT.2018.8639639) |

Place the extracted data under `data/grid/raw` and `data/chicagofswild/raw`.

## Entry points

```bash
python3 scripts/prepare_grid.py --help
python3 scripts/prepare_chicago.py --help
python3 scripts/extract_features.py --help
python3 scripts/train.py --config configs/grid.yaml
python3 scripts/train.py --config configs/chicagofswild.yaml
python3 scripts/evaluate.py --help
```
