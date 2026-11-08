"""Independent PyTorch path-gradient and region-blanking fixtures for browser math."""
import json,torch,numpy as np
from train import preprocess
from train_refined import ConvNet
m=json.load(open('1-conv-model.json'));g=json.load(open('2-gallery.json'));torch.set_num_threads(2);net=ConvNet();net.load_state_dict({k:torch.tensor(v) for k,v in m['weights'].items()});net.eval();out=[]
for e in g[::8]:
 x=torch.tensor(preprocess(e['waveform']));path=torch.stack([x*(k+.5)/64 for k in range(64)]).requires_grad_(True);l=net(path);(l[:,1]-l[:,0]).sum().backward();ig=(path.grad.mean(0)*x).numpy();margin=float((net(x[None])[0,1]-net(x[None])[0,0]).detach());occ=[]
 for k in range(9):
  xx=x.clone();xx[k*12:(k+1)*12]=0;l=net(xx[None]);occ.append(margin-float((l[0,1]-l[0,0]).detach()))
 out.append(dict(id=e['id'],integrated=ig.tolist(),occlusion=occ))
open('attribution-fixtures.json','w').write(json.dumps(out,separators=(',',':')))
