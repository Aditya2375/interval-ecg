import unittest,json
from pathlib import Path
import numpy as np,torch
from train import Net,preprocess,metrics,TEST,VAL
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.d=json.loads(Path('1-model.json').read_text())
 def test_splits(self):
  r=self.d['training']['records'];a,b,c=map(set,[r['train'],r['validation'],r['test']]);self.assertFalse(a&b or a&c or b&c);self.assertTrue({201,202}<=c)
 def test_preprocess(self):
  for e in self.d['examples']:np.testing.assert_allclose(preprocess(e['waveform']),e['input'],atol=1e-6)
 def test_inference(self):
  net=Net();net.load_state_dict({k:torch.tensor(v) for k,v in self.d['weights'].items()});net.eval()
  with torch.no_grad():
   for e in self.d['examples']:np.testing.assert_allclose(net(torch.tensor([e['input']])).softmax(-1)[0].numpy(),e['probabilities'],atol=1e-6)
 def test_metrics(self):
  z=np.load('test_predictions.npz');self.assertEqual(metrics(z['y'],z['p'].argmax(1))['confusion'],self.d['metrics']['test']['confusion'])
 def test_selected_epoch(self):
  h=self.d['training']['history'];self.assertEqual(min(h,key=lambda x:x['validation_loss'])['epoch'],self.d['training']['selected_epoch'])
 def test_hashes(self):
  import hashlib
  for name,h in self.d['source']['sha256'].items():self.assertEqual(hashlib.sha256((Path('data')/name).read_bytes()).hexdigest(),h)
if __name__=='__main__':unittest.main()
