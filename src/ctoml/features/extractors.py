
from __future__ import annotations
import hashlib
import sys
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def image_tensor(frames, size, grayscale=False):
    arrays = []
    for frame in frames:
        im = Image.fromarray(frame).convert('L' if grayscale else 'RGB').resize(size, Image.Resampling.BILINEAR)
        array = np.array(im, dtype=np.float32) / 255.0
        arrays.append(array[..., None] if grayscale else array)
    if not arrays:
        raise ValueError('No decoded frames')
    return torch.from_numpy(np.stack(arrays)).permute(0,3,1,2)


class ResNetExtractor:
    def __init__(self, name='resnet50', checkpoint=None, pretrained=False, device='cpu', batch_size=64):
        if name not in {'resnet18','resnet50'}:
            raise ValueError('Only resnet18 and resnet50 are supported')
        if bool(checkpoint) == bool(pretrained):
            raise ValueError('Provide exactly one local checkpoint or explicit --pretrained (may download)')
        if checkpoint and not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        from torchvision import models
        weights = getattr(models, 'ResNet18_Weights' if name == 'resnet18' else 'ResNet50_Weights').IMAGENET1K_V1 if pretrained else None
        model = getattr(models, name)(weights=weights)
        if checkpoint:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if 'state_dict' in state:
                state = state['state_dict']
            model.load_state_dict(state, strict=True)
        digest = sha256_file(checkpoint) if checkpoint else sha256_file(Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name)
        model.fc = torch.nn.Identity()
        self.model, self.device, self.batch_size = model.eval().to(device), device, batch_size
        self.metadata = dict(name=name, weights_sha256=digest, layer='avgpool', preprocessing=dict(color='RGB',size=[112,112],normalization='ImageNet'))

    @torch.no_grad()
    def __call__(self, frames):
        x = image_tensor(frames, (112,112))
        mean = x.new_tensor([.485,.456,.406])[None,:,None,None]
        std = x.new_tensor([.229,.224,.225])[None,:,None,None]
        x = (x-mean)/std
        return torch.cat([self.model(b.to(self.device)).cpu() for b in x.split(self.batch_size)]).numpy()


class AVHubertExtractor:
    def __init__(self, repo, checkpoint, output_layer=None, device='cpu'):
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError('AV-HuBERT requires a local trusted pretrained checkpoint')
        user_dir = Path(repo).resolve() / 'avhubert'
        if not (user_dir/'hubert.py').is_file():
            raise FileNotFoundError(user_dir / 'hubert.py')
        if output_layer is not None and output_layer < 1:
            raise ValueError('AV-HuBERT output layer is 1-based')
        try:
            from fairseq import checkpoint_utils, utils
        except ImportError as exc:
            raise RuntimeError('fairseq is required') from exc

        sys.path.insert(0, str(user_dir.parent))
        utils.import_user_module(Namespace(user_dir=str(user_dir)))
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)])
        model = models[0]
        if not hasattr(model, 'extract_finetune'):
            raise ValueError('Supply a pretrained AVHubertModel checkpoint, not a seq2seq wrapper')
        if output_layer is not None and output_layer > len(model.encoder.layers):
            raise ValueError('output_layer exceeds checkpoint encoder depth')
        self.model, self.device, self.layer = model.eval().to(device), device, output_layer
        self.image_size = int(getattr(cfg.task, 'image_crop_size', 88))
        self.image_mean = float(getattr(cfg.task, 'image_mean', .421))
        self.image_std = float(getattr(cfg.task, 'image_std', .165))
        if self.image_size < 1 or self.image_std <= 0:
            raise ValueError('Invalid image normalization in checkpoint')
        self.metadata = dict(name='avhubert', weights_sha256=sha256_file(checkpoint),
                             layer=output_layer or 'last', preprocessing=dict(color='grayscale',size=[self.image_size,self.image_size],
                             normalization=dict(mean=self.image_mean,std=self.image_std),
                             hubert_source_sha256=sha256_file(user_dir/'hubert.py'),
                             resize_policy='resize_100x60_mouth_to_checkpoint_crop_size'))

    @torch.no_grad()
    def __call__(self, frames):
        x = image_tensor(frames, (self.image_size,self.image_size), grayscale=True)
        x = ((x-self.image_mean)/self.image_std).permute(1,0,2,3).unsqueeze(0).to(self.device)
        padding = torch.zeros((1,x.shape[2]), dtype=torch.bool, device=self.device)


        features, mask = self.model.extract_finetune(source={'audio':None,'video':x},
                            padding_mask=padding, mask=False, output_layer=self.layer)
        if mask is not None:
            features = features[0][~mask[0]]
        else:
            features = features[0]
        return features.float().cpu().numpy()
