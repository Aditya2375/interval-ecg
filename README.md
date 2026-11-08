# Interval / 02

[Open the live signal laboratory](https://aditya2375.github.io/interval-ecg/)

A real, from-scratch 1D convolutional ECG morphology experiment with two local neural models, signed integrated gradients, region blanking, synthetic perturbations, threshold exploration and patient-group uncertainty. No hosted inference or runtime dependencies.

**Education only. Not a medical device, diagnosis, screening tool or health assessment. No personal ECG uploads.** N means a normal-beat annotation, not a healthy person. V means a premature ventricular contraction annotation. All other symbols are excluded, never mapped to normal. This is not the broader AAMI N/S/V/F/Q task.

## What changed

The original dense model remains unchanged in `1-model.json`: 108 → 64 ReLU → 2, 7,106 parameters, seed 812. The new `1-conv-model.json` has 15,994 parameters:

- Input: 108 normalized values, one channel.
- Convolution: 16 channels, kernel 7, zero padding 3; ReLU; average pooling by 2.
- Convolution: 24 channels, kernel 5, zero padding 2; ReLU; average pooling by 3.
- Channel-major flatten (24 × 18), dense 32 ReLU, two logits.

Seed 913. Fixed 12 epochs, AdamW learning rate 0.0005, weight decay 0.001, batch 256, gradient norm clip 5. Inverse-frequency class weights come only from training. Minimum class-weighted validation loss selected epoch 1. Validation contains only 90 V beats. No pretrained model, extra data, test-driven epoch or threshold selection. Two full reruns on the original training machine produced byte-identical checkpoint, audit and gallery files. This is machine-specific, not a cross-CPU guarantee. Independent reproduction with the same library versions on another CPU showed weight differences of roughly 1e-6 to 2e-4, differences in a few audit rows, and one stress macro F1 change of about 3e-5. In that reproduction, main-benchmark argmax predictions, headline metrics and the paired bootstrap reproduced. Score-vector checks against frozen weights allow a 1e-6 absolute/relative tolerance; new training runs may vary with CPU kernels.

## Evidence and its limits

The **previously inspected v1 test partition is reused as a development benchmark**. It is not an untouched new test or independent clinical validation. Training/validation remain separate from these records, but reusing a published benchmark does not restore its independence. No architecture search was run: this is one fixed convolutional experiment compared with the frozen original model.

24,577 eligible benchmark beats, fixed V-score threshold 0.50. Confusion rows are reference N/V, columns prediction N/V:

| Metric | Original dense | Convolutional |
|---|---:|---:|
| Confusion | [[20723, 482], [93, 3279]] | [[21108, 97], [125, 3247]] |
| Macro F1 | 0.95285256 | 0.98085672 |
| V precision | 0.87184260 | 0.97099282 |
| V recall | 0.97241993 | 0.96293001 |
| Accuracy | 0.97660414 | 0.99096716 |

N→V errors fall from 482 to 97, but missed V beats rise from 93 to 125. This is a tradeoff, not a clean win. Majority-N baseline accuracy is 86.27986%.

Paired patient-cluster bootstrap: macro F1 difference +0.02800416, 95% percentile interval **[-0.00557628, 0.07497819]**. 3,000 draws, seed 1203. Eleven groups, with records 201/202 combined because they belong to one person. Each draw samples groups with replacement and includes every beat from those sampled groups in both models' metrics. **The interval includes zero. Reliable patient-level improvement is not established.** A small historical, selected-label cohort does not establish clinical performance or prevalence.

`audit.json` includes every threshold from 0.05 to 0.95 in 0.05 steps, exact confusion counts and class metrics, every record, the paired interval and full-benchmark synthetic stress checks. Moving the threshold is exploratory; the app recommends none. Headline metrics stay at 0.50. Scores are not calibrated probabilities.

Stress checks run after training, with no model selection from the results: edge-replicated shifts of six original samples (16.7 ms) left/right; Gaussian noise at 0.10 and 0.25 times each original window's standard deviation, seed 1913. Every changed window is normalized again. These limited synthetic probes reveal substantial timing sensitivity; they are not real altered recordings or robustness guarantees. The UI's per-beat noise uses a separate fixed seed 1943 and a separate Box-Muller generator; it does not claim bit-for-bit identity with NumPy's full-cohort stress samples.

## Inspect rather than decorate

- 66 public examples: first two N and two V beats where available per benchmark record, first 16 dense/conv disagreements and first 12 convolutional errors, deduplicated. This deliberately selected gallery is not a prevalence sample. Its metadata and original float32 millivolt waveforms are in `2-gallery.json`.
- Every score is recalculated locally from the selected waveform. Original and normalized views, record/label/disagreement/error filters, empty states and navigation.
- Synthetic shifts up to ±18 original samples (50 ms) use edge replication. Fixed-seed Gaussian noise is scaled to the original window's standard deviation. The original annotation is not re-established for the altered input.
- Signed integrated gradients of the **V-minus-N logit margin**, not a medical explanation. A zero standardized-input baseline is a flat mathematical reference, not a normal heartbeat. The 64-step midpoint approximation reports margin, baseline, attribution sum and numerical completeness residual. Path points are in normalized feature space, not newly normalized raw ECG windows. Derivatives include the full convolution/pooling/dense network, not the raw preprocessing transform.
- Nine independent region probes blank 12 normalized inputs each (66.7 ms) to zero, without re-normalizing. Positive Δ means blanking lowered the V-minus-N margin. These may be out-of-distribution inputs. They are sensitivities, not proof of physiological causation.

## Source and partitions

Contains information from the [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/), Moody & Mark (2005), version 1.0.0, DOI 10.13026/C2F305, under the [Open Data Commons Attribution License v1.0](https://www.physionet.org/content/mitdb/view-license/1.0.0/). [Annotation definitions](https://physionet.org/physiotools/wpg/wpg_36.htm). All included source-file SHA-256 hashes are in both checkpoints. `train_refined.py` verifies the raw files against the original manifest before loading them.

Excluded paced records: 102, 104, 107, 217. Validation: 105, 111, 124, 209, 222. Benchmark: 101, 106, 114, 118, 200, 201, 202, 208, 213, 219, 223, 230. Other unpaced records train the models. 201/202 stay together. N/V counts: train 46,110/3,440; validation 7,207/90; benchmark 21,205/3,372. These are patient/record groups, not random beat splits. This is not a claim of following the classic DS1/DS2 split.

MLII only, selected by lead name (including record 114's different channel order). Expert annotations locate windows 72 samples before and 144 after the beat at 360 Hz. Only original N/V symbols with complete finite windows are retained. Pair averaging gives 108 samples, followed by per-window centering and standard deviation scaling with a 1e-6 floor. No added baseline filtering. Neighboring windows may overlap inside a partition, not across partitions. The model does not find beats in continuous signals.

## Reproduce

The live app needs no installation. To run a local copy, serve this directory with `python -m http.server 8000` and open localhost:8000. Training:

```sh
python -m pip install -r requirements.txt
python download.py
python train.py
python train_refined.py
python export_fixtures.py
python verify_audit.py
python -m unittest -v
node test_engine.cjs
node test_conv.cjs
```

Tested with PyTorch 2.14.1+cpu and wfdb 4.3.1, two CPU threads. Library/hardware changes can affect exact replay. Raw downloads are about 90 MB. Training creates `test_predictions.npz` and `paired_predictions.npz` for exhaustive metric tests; these local arrays are not needed by the browser and are not published. Files are JSON weights, never pickle. The .npz files contain numeric arrays only.

Eleven Python tests cover the original source hashes, preprocessing, grouping, baseline parity/metrics, both model selections, every threshold's confusion/metrics, every record and all gallery predictions/gradients. JavaScript tests compare all 66 convolutional predictions/gradients with PyTorch, central finite differences, nine independent PyTorch integrated-gradient and region-blanking fixtures, deterministic noise, zero-change identity and invalid input rejection. Maximum errors: probabilities 1.56e-7; gradient 8.50e-8; integrated gradients 2.22e-7; region deltas 7.41e-7.

Browser harness: `npm install --no-save playwright`, then `node browser-check.cjs`. Chrome defaults to `/usr/bin/google-chrome`, override with `CHROME_PATH`. Screenshots save to `/tmp` or `SCREENSHOT_DIR`. `BASE_URL` tests a deployed URL instead of a local server. Tests cover both models, gallery boundaries, disagreements, raw/normalized view, shift/noise/reset, real attribution/occlusion, extreme thresholds, fixed headline metrics, empty filters, 320/390px overflow and runtime errors.

Method references: [Integrated Gradients, Sundararajan et al.](https://arxiv.org/html/1703.01365v2); [Conv1d](https://docs.pytorch.org/docs/main/generated/torch.nn.Conv1d.html); [AvgPool1d](https://docs.pytorch.org/docs/main/generated/torch.nn.AvgPool1d.html). Interpatient evaluation context: [Wang et al. (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8181174/). This app does not reproduce that paper's architecture, labels, split or clinical claims.

No accounts, uploads, cookies, storage, telemetry, remote inference, dynamic code or third-party runtime scripts. Same-origin CSP, safe DOM text, validated finite inputs. Historical recordings, selected arrhythmias, only two labels, small subject count, no external cohort and no clinical validation. No deployment or health assessment claim.
