"""Educational N-vs-V annotated-beat classification, not a clinical device."""
import json,hashlib,copy
from pathlib import Path
import numpy as np,torch,wfdb
from torch import nn
TRAIN_EXCLUDE={102,104,107,217}
VAL={105,111,124,209,222}
TEST={101,106,114,118,200,201,202,208,213,219,223,230}
class Net(nn.Module):
 def __init__(self):super().__init__();self.fc=nn.Linear(108,64);self.head=nn.Linear(64,2)
 def forward(self,x):return self.head(torch.relu(self.fc(x)))
def preprocess(x):
 x=np.asarray(x,dtype=np.float32).reshape(108,2).mean(1);return ((x-x.mean())/max(float(x.std()),1e-6)).astype(np.float32)
def metrics(y,p):
 c=np.zeros((2,2),int)
 for a,b in zip(y,p):c[a,b]+=1
 rows=[]
 for k in range(2):
  tp=c[k,k];precision=float(tp/max(c[:,k].sum(),1));recall=float(tp/max(c[k,:].sum(),1));rows.append(dict(label=['N','V'][k],support=int(c[k].sum()),precision=precision,recall=recall,f1=2*precision*recall/max(precision+recall,1e-9)))
 return dict(confusion=c.tolist(),accuracy=float(np.mean(y==p)),classes=rows,macro_f1=float(np.mean([r['f1'] for r in rows])))
def main():
 torch.set_num_threads(2);torch.manual_seed(812);rng=np.random.default_rng(812);data={k:[] for k in ['train','validation','test']};hashes={};excluded={};records={k:[] for k in data}
 for file in sorted(Path('data').glob('*.hea')):
  rec=int(file.stem)
  if rec in TRAIN_EXCLUDE:continue
  split='validation' if rec in VAL else 'test' if rec in TEST else 'train';records[split].append(rec);record=wfdb.rdrecord(str(file.with_suffix('')));ann=wfdb.rdann(str(file.with_suffix('')),'atr');lead=record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
  if record.sig_name[lead]!='MLII':raise ValueError('Expected MLII')
  for ext in ['dat','hea','atr']:
   p=file.with_suffix('.'+ext);hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
  for sample,sym in zip(ann.sample,ann.symbol):
   if sym not in ['N','V']:excluded[sym]=excluded.get(sym,0)+1;continue
   if sample<72 or sample+144>len(record.p_signal):continue
   x=record.p_signal[sample-72:sample+144,lead]
   if not np.isfinite(x).all():continue
   data[split].append((preprocess(x),int(sym=='V'),rec,int(sample),x.astype(np.float32)))
 arrays={k:(torch.tensor(np.stack([r[0] for r in rows])),torch.tensor([r[1] for r in rows])) for k,rows in data.items()};net=Net();optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001);counts=np.bincount(arrays['train'][1].numpy(),minlength=2);weight=torch.tensor(counts.sum()/(2*counts),dtype=torch.float32);history=[];best=1e9;state=None
 for epoch in range(1,21):
  net.train();perm=torch.randperm(len(arrays['train'][0]));losses=[]
  for ids in perm.split(256):
   x,y=arrays['train'];loss=nn.functional.cross_entropy(net(x[ids]),y[ids],weight=weight);optim.zero_grad();loss.backward();optim.step();losses.append(loss.item())
  net.eval()
  with torch.no_grad():x,y=arrays['validation'];val=nn.functional.cross_entropy(net(x),y,weight=weight).item()
  history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=val));print(history[-1],flush=True)
  if val<best:best=val;state=copy.deepcopy(net.state_dict());bestepoch=epoch
 net.load_state_dict(state);net.eval();reports={};perrecord={}
 with torch.no_grad():
  for split,(x,y) in arrays.items():
   probs=net(x).softmax(-1).numpy();pred=probs.argmax(1);reports[split]=metrics(y.numpy(),pred)
   if split=='test':
    testprobs=probs
    for rec in records['test']:
     ids=np.array([i for i,r in enumerate(data['test']) if r[2]==rec]);perrecord[str(rec)]=metrics(y.numpy()[ids],pred[ids])
 # deterministic, label-stratified examples plus the first errors. Not a prevalence sample.
 test=data['test'];ys=np.array([r[1] for r in test]);ids=[]
 for k in [0,1]:ids.extend(rng.choice(np.where(ys==k)[0],size=12,replace=False).tolist())
 ids+=np.where(testprobs.argmax(1)!=ys)[0][:8].tolist();ids=list(dict.fromkeys(ids));examples=[]
 for i in ids:
  x,y,rec,sample,raw=test[i];examples.append(dict(id=f'{rec}-{sample}',record=rec,sample=sample,time_seconds=sample/360,label=['N','V'][y],waveform=raw.tolist(),input=x.tolist(),probabilities=testprobs[i].tolist()))
 payload=dict(version=1,config=dict(input=108,hidden=64,classes=['N','V'],sampling_hz=360,window_before=72,window_after=144),weights={k:v.tolist() for k,v in state.items()},training=dict(seed=812,epochs=20,selected_epoch=bestepoch,history=history,class_weights=weight.tolist(),counts={k:np.bincount(arrays[k][1].numpy(),minlength=2).tolist() for k in arrays},records=records,excluded_symbols=excluded,excluded_paced_records=sorted(TRAIN_EXCLUDE),parameters=sum(p.numel() for p in net.parameters())),metrics=reports,per_record_test=perrecord,examples=examples,source=dict(url='https://www.physionet.org/content/mitdb/1.0.0/',license='Open Data Commons Attribution License v1.0',sha256=hashes))
 Path('1-model.json').write_text(json.dumps(payload,separators=(',',':')));np.savez_compressed('test_predictions.npz',y=ys,p=testprobs,record=np.array([r[2] for r in test]));print('TEST',reports['test'],'selected',bestepoch,flush=True)
if __name__=='__main__':main()
