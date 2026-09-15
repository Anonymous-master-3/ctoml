import csv
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from ctoml.features.manifests import grid_records, chicago_records, GRID_TEST_SPEAKERS
from ctoml.features.preprocess import mouth_crop, face_roi, read_frames
from ctoml.features.extractors import image_tensor, ResNetExtractor, AVHubertExtractor
from ctoml.features.pipeline import extract_manifest


def test_grid_split_identity_and_alignment(tmp_path):
    people = set().union(*GRID_TEST_SPEAKERS) | {'s9','s10'}
    for person in people:
        folder = tmp_path/person
        folder.mkdir()
        (folder/'a.mpg').touch()
        (folder/'a.align').write_text('0 10 sil\n10 20 bin\n20 30 blue\n30 40 sp\n')
    for split in range(1,5):
        rows = grid_records(tmp_path,tmp_path,split,['s9'])
        assert {r['performer_id'] for r in rows if r['split']=='test'} == GRID_TEST_SPEAKERS[split-1]
        assert {r['performer_id'] for r in rows if r['split']=='dev'} == {'s9'}
        assert all(r['transcript']=='bin blue' for r in rows)
    with pytest.raises(ValueError,match='disjoint'):
        grid_records(tmp_path,tmp_path,1,['s1'])


def test_chicago_csv_preserves_labels_and_rejects_leak(tmp_path):
    (tmp_path/'a').mkdir()
    (tmp_path/'b').mkdir()
    path = tmp_path/'meta.csv'
    path.write_text('filename,number_of_frames,label_proc,partition,signer\na,2,a&b,train,one\nb,3,Ab!,dev,two\n')
    rows = chicago_records(path,tmp_path)
    assert rows[0]['transcript']=='a&b'
    assert rows[1]['transcript']=='Ab!'
    path.write_text(path.read_text().replace('two','one'))
    with pytest.raises(ValueError,match='leakage'):
        chicago_records(path,tmp_path)


def test_roi_geometry_padding_and_color():
    frame = np.full((100,160,3), [255,0,0],dtype=np.uint8)
    landmarks = np.tile([10,10],(68,1))
    mouth = mouth_crop(frame,landmarks)
    assert mouth.shape==(60,100,3)
    assert np.all(mouth[:20]==0)
    assert np.all(mouth[30,50]==[255,0,0])
    roi = face_roi(frame,[20,20,40,40])
    assert roi.shape==(112,112,3)
    rgb = image_tensor([frame],(112,112))
    assert torch.equal(rgb[0,:,50,50],torch.tensor([1.,0.,0.]))
    gray = image_tensor([frame],(88,88),grayscale=True)
    assert gray.shape==(1,1,88,88)
    assert gray[0,0,0,0].item()==pytest.approx(76/255)


def test_frame_order_is_numeric(tmp_path):
    for name,value in [('10.png',100),('2.png',20),('1.png',10)]:
        Image.fromarray(np.full((3,3,3),value,dtype=np.uint8)).save(tmp_path/name)
    assert [int(f[0,0,0]) for f in read_frames(tmp_path)]==[10,20,100]


def test_extract_cache_source_and_flip(tmp_path):
    source = tmp_path/'frames'
    source.mkdir()
    frame = np.zeros((8,8,3),dtype=np.uint8)
    frame[:,:4]=255
    Image.fromarray(frame).save(source/'1.png')
    manifest = tmp_path/'raw.jsonl'
    manifest.write_text(json.dumps(dict(sample_id='a',performer_id='s',domain_id='s',split='train',
                                       video_path='frames',transcript='a',num_frames=1)))
    class Fixture:
        metadata = dict(name='fixture',weights_sha256=None,layer='fixture',preprocessing={})
        count=0
        def __call__(self,frames):
            self.count+=1

            return np.array([[f[:,0].mean(),f[:,-1].mean()] for f in frames],dtype=np.float32)
    fixture=Fixture()
    def run():
        return extract_manifest(manifest,tmp_path/'out.jsonl',tmp_path/'cache',fixture,cache_flips=True)[0]
    first=run()
    assert fixture.count==2
    assert np.load(first['feature_path']).tolist()==[[255.,0.]]
    assert np.load(first['feature_path_flipped']).tolist()==[[0.,255.]]
    run()
    assert fixture.count==2
    Image.fromarray(np.full((8,8,3),80,dtype=np.uint8)).save(source/'1.png')
    second=run()
    assert second['feature_path'] != first['feature_path']
    assert fixture.count==4


def test_weight_requirements_fail_before_optional_import():
    with pytest.raises(ValueError,match='exactly one'):
        ResNetExtractor()
    with pytest.raises(FileNotFoundError):
        AVHubertExtractor('/missing/repo',None)


def test_avhubert_pure_video_interface_fixture():
    class Fake(torch.nn.Module):
        def extract_finetune(self,source,padding_mask,mask,output_layer):
            assert source['audio'] is None
            assert source['video'].shape==(1,1,2,88,88)
            assert padding_mask.shape==(1,2) and not padding_mask.any()
            assert mask is False and output_layer==3
            return source['video'].mean((-1,-2)).transpose(1,2),padding_mask
    extractor=AVHubertExtractor.__new__(AVHubertExtractor)
    extractor.model,extractor.device,extractor.layer=Fake(),'cpu',3
    extractor.image_size,extractor.image_mean,extractor.image_std=88,.421,.165
    features=extractor([np.zeros((60,100,3),dtype=np.uint8)]*2)
    assert features.shape==(2,1)
    assert np.allclose(features,-.421/.165)


def test_resnet_local_checkpoint_and_batching_fixture(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc=torch.nn.Linear(3,2)
            self.dropout=torch.nn.Dropout(.9)
        def forward(self,x):
            return self.fc(self.dropout(x.mean((-1,-2))))
    checkpoint=tmp_path/'model.pt'
    torch.save(Backbone().state_dict(),checkpoint)
    calls=[]
    def factory(weights):
        calls.append(weights)
        return Backbone()
    monkeypatch.setitem(sys.modules,'torchvision',SimpleNamespace(models=SimpleNamespace(resnet18=factory)))
    extractor=ResNetExtractor('resnet18',checkpoint=checkpoint,batch_size=1)
    features=extractor([np.full((10,20,3),255,dtype=np.uint8)]*3)
    assert calls==[None]
    assert not extractor.model.training
    assert features.shape==(3,3)
    assert np.allclose(features[0],(1-np.array([.485,.456,.406]))/np.array([.229,.224,.225]),atol=1e-5)
    assert len(extractor.metadata['weights_sha256'])==64


def test_reextract_drops_stale_flip_and_uses_source_count(tmp_path):
    source=tmp_path/'frames'
    source.mkdir()
    for i in range(4):
        Image.fromarray(np.full((4,4,3),i*30,dtype=np.uint8)).save(source/(str(i)+'.png'))
    manifest=tmp_path/'raw.jsonl'
    manifest.write_text(json.dumps(dict(sample_id='a',performer_id='s',domain_id='s',split='train',
                                       video_path='frames',transcript='a',num_frames=4)))
    class Downsample:
        metadata=dict(name='downsample_fixture',weights_sha256=None,layer='fixture',preprocessing={})
        def __call__(self,frames):
            return np.ones((len(frames)//2,3),dtype=np.float32)
    first=extract_manifest(manifest,tmp_path/'old.jsonl',tmp_path/'cache',Downsample(),cache_flips=True)[0]
    assert first['num_frames']==2 and first['source_num_frames']==4
    assert 'feature_path_flipped' in first
    class NewEncoder:
        metadata=dict(name='new_encoder_fixture',weights_sha256=None,layer='fixture',preprocessing={})
        def __call__(self,frames):
            return np.ones((len(frames),5),dtype=np.float32)
    second=extract_manifest(tmp_path/'old.jsonl',tmp_path/'new.jsonl',tmp_path/'cache',NewEncoder())[0]
    assert second['num_frames']==4 and second['source_num_frames']==4 and second['feature_dim']==5
    assert 'feature_path_flipped' not in second
    assert second['feature_path']!=first['feature_path']
    first['source_num_frames']=3
    (tmp_path/'bad.jsonl').write_text(json.dumps(first))
    with pytest.raises(ValueError,match='frame count mismatch'):
        extract_manifest(tmp_path/'bad.jsonl',tmp_path/'badout.jsonl',tmp_path/'cache',NewEncoder())
