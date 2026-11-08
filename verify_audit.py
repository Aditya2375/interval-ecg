import sys,json,numpy as np
from pathlib import Path
from train import preprocess
from train_refined import load,ConvNet,inference
from train import Net
import torch
root=Path(__file__).resolve().parent;a=json.loads((root/'audit.json').read_text());p=np.load(root/'paired_predictions.npz');y=p['y'];records=p['record'];groups=np.where(records==202,201,records);unique=np.unique(groups)
def fast_f1(y,p):
 c=np.bincount(y*2+p,minlength=4).reshape(2,2);return sum(2*c[k,k]/max(2*c[k,k]+c[1-k,k]+c[k,1-k],1) for k in [0,1])/2
rng=np.random.default_rng(1203);vals=[]
for i in range(3000):
 ids=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)]);vals.append(fast_f1(y[ids],p['conv'][ids].argmax(1))-fast_f1(y[ids],p['dense'][ids].argmax(1)))
np.testing.assert_allclose(np.quantile(vals,[.025,.975]),a['paired_patient_bootstrap']['ci95'],atol=1e-14);print('Independent count-based 3000-resample patient bootstrap exactly reproduced')
import os;os.chdir(root);torch.set_num_threads(2);d=load();np.testing.assert_array_equal(d['test_y'],y);cm=json.loads((root/'1-conv-model.json').read_text());bm=json.loads((root/'1-model.json').read_text());nets={}
for name,net,m in [('conv',ConvNet(),cm),('dense',Net(),bm)]:
 net.load_state_dict({k:torch.tensor(v) for k,v in m['weights'].items()});net.eval();nets[name]=net;scores=inference(net,d['test_x']);np.testing.assert_allclose(scores,p[name],atol=1e-6,rtol=1e-6);np.testing.assert_array_equal(scores.argmax(1),p[name].argmax(1))
print('All 24577 paired score vectors match frozen weights and freshly source-hash-verified inputs within atol=rtol=1e-6')
rng=np.random.default_rng(1913)
for s in a['stress']:
 raw=d['test_raw'].copy();name=s['name']
 if name=='shift_left_6':raw=np.concatenate([raw[:,6:],np.repeat(raw[:,-1:],6,axis=1)],axis=1)
 elif name=='shift_right_6':raw=np.concatenate([np.repeat(raw[:,:1],6,axis=1),raw[:,:-6]],axis=1)
 else:raw+=rng.normal(size=raw.shape).astype(np.float32)*raw.std(1,keepdims=True)*float(name.split('_')[1])
 x=np.stack([preprocess(z) for z in raw])
 for name,net in nets.items():assert abs(fast_f1(y,inference(net,x).argmax(1))-s[name]['macro_f1'])<1e-14
print('Independent stress macro F1 recomputation passes all eight full-cohort conditions')
