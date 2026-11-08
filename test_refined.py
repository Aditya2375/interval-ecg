import unittest,json,numpy as np,torch
from pathlib import Path
from train import metrics
from train_refined import ConvNet
class RefinedChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);cls.m=json.loads(Path('1-conv-model.json').read_text());cls.a=json.loads(Path('audit.json').read_text());cls.g=json.loads(Path('2-gallery.json').read_text());cls.p=np.load('paired_predictions.npz');cls.net=ConvNet();cls.net.load_state_dict({k:torch.tensor(v) for k,v in cls.m['weights'].items()});cls.net.eval()
 def test_weights_and_selection(self):
  self.assertEqual(sum(p.numel() for p in self.net.parameters()),15994);t=self.m['training'];self.assertEqual(len(t['history']),12);self.assertEqual(min(t['history'],key=lambda h:h['validation_loss'])['epoch'],t['selected_epoch']);self.assertEqual(t['records'],json.loads(Path('1-model.json').read_text())['training']['records'])
 def test_all_metric_thresholds(self):
  y=self.p['y']
  for t in self.a['thresholds']:
   for name in ['dense','conv']:self.assertEqual(metrics(y,(self.p[name][:,1]>=t['threshold']).astype(int)),t[name])
  for name in ['dense','conv']:self.assertEqual(metrics(y,self.p[name].argmax(1)),self.a[name])
 def test_per_record(self):
  for r,m in self.a['per_record'].items():
   ids=self.p['record']==int(r)
   for name in ['dense','conv']:self.assertEqual(metrics(self.p['y'][ids],self.p[name][ids].argmax(1)),m[name])
 def test_gallery_prediction_gradient(self):
  from train import preprocess
  for e in self.g:
   x=torch.tensor(preprocess(e['waveform'])[None],requires_grad=True);l=self.net(x);p=l.softmax(-1).detach().numpy()[0];margin=l[0,1]-l[0,0];margin.backward();np.testing.assert_allclose(p,e['conv_probabilities'],atol=2e-6);np.testing.assert_allclose(x.grad.numpy()[0],e['gradient'],atol=2e-6);self.assertAlmostEqual(float(margin.detach()),e['margin'],places=5)
 def test_cluster_interval(self):
  b=self.a['paired_patient_bootstrap'];self.assertNotIn(202,b['groups']);self.assertIn(201,b['groups']);self.assertEqual(len(b['groups']),11);self.assertLess(b['ci95'][0],0);self.assertGreater(b['ci95'][1],0)
if __name__=='__main__':unittest.main()
