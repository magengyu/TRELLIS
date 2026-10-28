"""Small CPU regression tests; no downloads or original repository imports."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch

from ss_module.models import build_model, load_model
from ss_module.pipelines import OccupancyPipeline
from ss_module.pipelines.samplers import sample_flow
from ss_module.trainers import train

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def config(stage):
    return json.loads((ROOT/'configs'/f'{stage}_tiny.json').read_text())


def dataset(root):
    root.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(1)
    records=[]
    for i in range(3):
        grid=(rng.random((8,8,8))>.8)
        np.savez_compressed(root/f'{i}.npz',occupancy=grid,
                            latent=rng.normal(size=(2,4,4,4)).astype('float32'),
                            cond=rng.normal(size=(2,5,12)).astype('float32'))
        records.append(dict(occupancy=f'{i}.npz',latent=f'{i}.npz',cond=f'{i}.npz'))
    manifest=root/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    return manifest


@pytest.mark.parametrize('stage',['vae','flow'])
def test_train_resume_and_export(tmp_path,stage):
    cfg=config(stage);cfg['ema_decay']=.9;cfg['accumulation_steps']=2
    manifest=dataset(tmp_path/'data')
    full=train(cfg,manifest,tmp_path/'full',device='cpu')
    first=copy.deepcopy(cfg);first['max_steps']=1
    checkpoint=train(first,manifest,tmp_path/'split',device='cpu')
    resumed=train(cfg,manifest,tmp_path/'split',device='cpu',resume=checkpoint)
    a=torch.load(full,weights_only=False,map_location='cpu')
    b=torch.load(resumed,weights_only=False,map_location='cpu')
    assert a['step']==b['step']==2
    for role in a['models']:
        for key in a['models'][role]:
            torch.testing.assert_close(a['models'][role][key],b['models'][role][key],rtol=0,atol=0)
            torch.testing.assert_close(a['ema'][role][key],b['ema'][role][key],rtol=0,atol=0)
        model=load_model(tmp_path/'split'/'weights'/role)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,b['models'][role][key],rtol=0,atol=0)


def test_vae_roundtrip_and_flow_modes(tmp_path):
    cfg=config('vae');cfg['max_steps']=1
    manifest=dataset(tmp_path/'data')
    train(cfg,manifest,tmp_path/'vae',device='cpu')
    weights=tmp_path/'vae'/'weights'
    encoder=load_model(weights/'encoder');decoder=load_model(weights/'decoder')
    with torch.no_grad():
        z,mean,logvar=encoder(torch.zeros(1,1,8,8,8),return_raw=True)
        assert z.shape==mean.shape==logvar.shape==(1,2,4,4,4)
        assert decoder(z).shape==(1,1,8,8,8)
    cfg=config('flow');cfg['max_steps']=1
    train(cfg,manifest,tmp_path/'flow',device='cpu')
    pipe=OccupancyPipeline(tmp_path/'flow'/'weights',device='cpu',decoder_weights=weights)
    cond=torch.randn(2,5,12)
    for mode in ('stochastic','multidiffusion'):
        first=pipe.generate(cond=cond,steps=2,seed=4,mode=mode)
        second=pipe.generate(cond=cond,steps=2,seed=4,mode=mode)
        assert first['occupancy'].shape==(8,8,8)
        np.testing.assert_array_equal(first['logits'],second['logits'])
    with pytest.raises(ValueError,match='at least one step'):
        pipe.generate(cond=cond,steps=1)


def test_sampling_euler_cfg_contract():
    class Model(torch.nn.Module):
        resolution=1;in_channels=1;cond_channels=1;device=torch.device('cpu')
        def forward(self,x,t,cond):
            return torch.ones_like(x)*cond[:,0,0,None,None,None,None]
    model=Model();cond=torch.tensor([[[2.]],[[4.]]])
    base=sample_flow(model,cond,steps=2,seed=3,cfg_strength=0,rescale_t=1,verbose=False)
    guided=sample_flow(model,cond,steps=2,seed=3,cfg_strength=1,cfg_interval=(0,1),rescale_t=1,verbose=False)
    torch.testing.assert_close(guided,base-3)


def test_no_repository_or_sparse_imports():
    code='''
import sys
class Deny:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'trellis','spconv','flash_attn','xformers','utils3d','nvdiffrast','kaolin'}:
            raise RuntimeError('Forbidden dependency: '+fullname)
sys.meta_path.insert(0,Deny())
from ss_module.pipelines import OccupancyPipeline
from ss_module.trainers import train
from ss_module.prepare import main
print('standalone imports passed')
'''
    subprocess.run([sys.executable,'-c',code],cwd=ROOT.parent,check=True)
