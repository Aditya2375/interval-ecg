"""Fixed convolutional experiment. Same previously inspected development benchmark, not new clinical evidence."""
import json,copy,hashlib
from pathlib import Path
import numpy as np,torch,wfdb
from torch import nn
from train import Net,preprocess,metrics,TRAIN_EXCLUDE,VAL,TEST
class ConvNet(nn.Module):
 def __init__(self):
  super().__init__();self.c1=nn.Conv1d(1,16,7,padding=3);self.c2=nn.Conv1d(16,24,5,padding=2);self.fc=nn.Linear(24*18,32);self.head=nn.Linear(32,2)
 def forward(self,x):
  x=nn.functional.avg_pool1d(torch.relu(self.c1(x[:,None])),2);x=nn.functional.avg_pool1d(torch.relu(self.c2(x)),3);return self.head(torch.relu(self.fc(x.flatten(1))))
def load():
 source=json.load(open('1-model.json'))['source']['sha256']
 for name,expected in source.items():
  if hashlib.sha256((Path('data')/name).read_bytes()).hexdigest()!=expected:raise ValueError('Source hash mismatch: '+name)
 out={}
 for split in ['train','validation','test']:
  rows=[]
  for file in sorted(Path('data').glob('*.hea')):
   rec=int(file.stem)
   if rec in TRAIN_EXCLUDE:continue
   part='validation' if rec in VAL else 'test' if rec in TEST else 'train'
   if part!=split:continue
   r=wfdb.rdrecord(str(file.with_suffix('')));a=wfdb.rdann(str(file.with_suffix('')),'atr');lead=r.sig_name.index('MLII')
   for sample,sym in zip(a.sample,a.symbol):
    if sym not in ['N','V'] or sample<72 or sample+144>len(r.p_signal):continue
    raw=r.p_signal[sample-72:sample+144,lead].astype(np.float32)
    if np.isfinite(raw).all():rows.append((preprocess(raw),int(sym=='V'),rec,int(sample),raw))
  for i,k in enumerate(['x','y','record','sample','raw']):out[split+'_'+k]=np.array([r[i] for r in rows])
 return out
def inference(net,x):
 with torch.no_grad():return torch.cat([net(z).softmax(-1) for z in torch.tensor(x).split(1024)]).numpy()
def main():
 torch.set_num_threads(2);torch.manual_seed(913);d=load();net=ConvNet();o=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.001);counts=np.bincount(d['train_y'],minlength=2);w=torch.tensor(counts.sum()/(2*counts),dtype=torch.float32);best=float('inf');history=[]
 x=torch.tensor(d['train_x']);y=torch.tensor(d['train_y']);vx=torch.tensor(d['validation_x']);vy=torch.tensor(d['validation_y'])
 for epoch in range(1,13):
  net.train();losses=[]
  for ids in torch.randperm(len(x)).split(256):
   loss=nn.functional.cross_entropy(net(x[ids]),y[ids],weight=w);o.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);o.step();losses.append(loss.item())
  net.eval()
  with torch.no_grad():vl=nn.functional.cross_entropy(net(vx),vy,weight=w).item()
  history.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation_loss=vl));print(history[-1],flush=True)
  if vl<best:best=vl;state=copy.deepcopy(net.state_dict());selected=epoch
 net.load_state_dict(state);net.eval();base=json.load(open('1-model.json'));old=Net();old.load_state_dict({k:torch.tensor(v) for k,v in base['weights'].items()});old.eval();p=inference(net,d['test_x']);bp=inference(old,d['test_x']);ys=d['test_y'];rec=d['test_record'];groups=np.where(rec==202,201,rec);unique=np.unique(groups)
 # Patient-cluster bootstrap; comparison is a development benchmark already exposed by v1.
 rng=np.random.default_rng(1203);boot=[]
 for _ in range(3000):
  draw=rng.choice(unique,len(unique),replace=True);ids=np.concatenate([np.flatnonzero(groups==g) for g in draw]);boot.append(metrics(ys[ids],p[ids].argmax(1))['macro_f1']-metrics(ys[ids],bp[ids].argmax(1))['macro_f1'])
 def threshold_report(pr,t,ids=None):
  if ids is None:ids=np.arange(len(ys))
  return metrics(ys[ids],(pr[ids,1]>=t).astype(int))
 thresholds=[]
 for t in np.arange(.05,1,.05):thresholds.append(dict(threshold=round(float(t),2),dense=threshold_report(bp,t),conv=threshold_report(p,t)))
 perrecord={str(r):dict(dense=threshold_report(bp,.5,np.flatnonzero(rec==r)),conv=threshold_report(p,.5,np.flatnonzero(rec==r))) for r in np.unique(rec)}
 report=dict(split_note='Previously inspected v1 test partition reused as a development benchmark. Not an untouched new test or clinical validation.',thresholds=thresholds,per_record=perrecord,dense=metrics(ys,bp.argmax(1)),conv=metrics(ys,p.argmax(1)),paired_patient_bootstrap=dict(groups=unique.tolist(),resamples=3000,seed=1203,metric='macro_f1 difference conv minus dense',estimate=metrics(ys,p.argmax(1))['macro_f1']-metrics(ys,bp.argmax(1))['macro_f1'],ci95=np.quantile(boot,[.025,.975]).tolist()),disagreements=int(np.sum(bp.argmax(1)!=p.argmax(1))))
 # Post-training synthetic stress tests. No retraining or model selection from these outcomes.
 stress=[];rng=np.random.default_rng(1913)
 for name in ['shift_left_6','shift_right_6','noise_0.10','noise_0.25']:
  raw=d['test_raw'].copy()
  if name.startswith('shift'):
   if 'left' in name:raw=np.concatenate([raw[:,6:],np.repeat(raw[:,-1:],6,axis=1)],axis=1)
   else:raw=np.concatenate([np.repeat(raw[:,:1],6,axis=1),raw[:,:-6]],axis=1)
  else:raw+=rng.normal(size=raw.shape).astype(np.float32)*raw.std(1,keepdims=True)*float(name.split('_')[1])
  xx=np.stack([preprocess(r) for r in raw]);stress.append(dict(name=name,dense=metrics(ys,inference(old,xx).argmax(1)),conv=metrics(ys,inference(net,xx).argmax(1))))
 report['stress']=stress
 # Diverse gallery: first N and V per record, then first disagreements and mistakes (deduplicated).
 ids=[]
 for r in np.unique(rec):
  for k in [0,1]:
   found=np.flatnonzero((rec==r)&(ys==k));ids.extend(found[:2].tolist())
 ids+=np.flatnonzero(bp.argmax(1)!=p.argmax(1))[:16].tolist();ids+=np.flatnonzero(p.argmax(1)!=ys)[:12].tolist();ids=list(dict.fromkeys(ids));gallery=[]
 for i in ids:
  inp=torch.tensor(d['test_x'][i:i+1],requires_grad=True);logits=net(inp);margin=(logits[0,1]-logits[0,0]);margin.backward();g=inp.grad.numpy()[0]
  gallery.append(dict(id=f'{rec[i]}-{d["test_sample"][i]}',record=int(rec[i]),sample=int(d['test_sample'][i]),label=['N','V'][ys[i]],waveform=d['test_raw'][i].tolist(),dense_probabilities=bp[i].tolist(),conv_probabilities=p[i].tolist(),margin=float(margin.detach()),gradient=g.tolist()))
 payload=dict(version=2,architecture='108 → Conv16×7 → AvgPool2 → Conv24×5 → AvgPool3 → Dense32 → 2',weights={k:v.tolist() for k,v in state.items()},training=dict(seed=913,epochs=12,selected_epoch=selected,lr=.0005,batch=256,weight_decay=.001,gradient_clip=5,counts=base['training']['counts'],records=base['training']['records'],history=history,parameters=sum(z.numel() for z in net.parameters()),class_weights=w.tolist()),source=base['source'])
 Path('1-conv-model.json').write_text(json.dumps(payload,separators=(',',':')));Path('audit.json').write_text(json.dumps(report,separators=(',',':')));Path('2-gallery.json').write_text(json.dumps(gallery,separators=(',',':')));np.savez_compressed('paired_predictions.npz',y=ys,dense=bp,conv=p,record=rec,sample=d['test_sample']);print('RESULT',report['dense'],report['conv'],report['paired_patient_bootstrap'],flush=True)
if __name__=='__main__':main()
