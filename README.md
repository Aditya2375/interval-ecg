# Interval

Educational two-label ECG morphology classifier. Not a medical device, diagnosis, screening tool or health assessment. No personal ECG uploads. N is a normal-beat annotation, not a healthy person; V is a premature ventricular contraction annotation. All other symbols are excluded, not mapped to normal. This is not the broader AAMI N/S/V/F/Q task.

Contains information from the [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/), made available under the [Open Data Commons Attribution License v1.0](https://www.physionet.org/content/mitdb/view-license/1.0.0/). Moody, G., & Mark, R. (2005), version 1.0.0, DOI 10.13026/C2F305. Waveform examples are derived from this database; original samples, source hashes and attribution are included in 1-model.json. [Annotation definitions](https://physionet.org/physiotools/wpg/wpg_36.htm).

## Run and reproduce

Run `python -m http.server 8000`, then open http://localhost:8000 for the static app. To reproduce: `python -m pip install -r requirements.txt`, `python download.py`, `python train.py`, `python -m unittest -v`, `node test_engine.cjs`. Raw data downloads are about 90 MB. Tested with PyTorch 2.14.1+cpu and wfdb 4.3.1; library or CPU changes may affect exact training replay. All original file SHA-256 hashes are embedded. No pretrained weights or hosted inference.

Excluded paced records: 102, 104, 107 and 217. Test records: 101, 106, 114, 118, 200, 201, 202, 208, 213, 219, 223 and 230. Validation records: 105, 111, 124, 209 and 222. All other records are used for training. The source says records 201 and 202 are the same subject, and both are assigned to test. Splits are record/patient groups, not random beats. MLII lead only; record 114's different channel order is handled by lead name. Only original N/V symbols with a complete 216-sample window are retained. No additional baseline filtering. Expert annotation locates each beat: the model does not detect beats in continuous ECG.

Window: 72 samples before and 144 after the annotation at 360 Hz. Pairwise mean downsampling produces 108 values. Each window is normalized by its own mean and standard deviation, with a 1e-6 floor. Dense network: 108 inputs, 64 ReLU units, 2 outputs, 7,106 parameters. Seed 812; 20 epochs; batch 256; AdamW learning rate 0.001, weight decay 0.001; inverse-frequency class weights from training only. Minimum class-weighted validation loss selected epoch 1. Test was used once after selection. Majority-N baseline and per-record/class metrics are exposed; no calibration or clinical validation claim.

Test: 24,577 beats. Confusion matrix rows are reference N/V and columns are prediction N/V: [[20723, 482], [93, 3279]]. V recall: 0.97242; V precision: 0.87184; overall accuracy: 0.97660; macro F1: 0.95285. This narrowly filtered historical task does not estimate performance or prevalence in a clinical population. Adjacent beat windows may overlap within records, not across partitions.

The gallery contains 12 sampled N beats, 12 sampled V beats, and the first 8 held-out errors. It is not a prevalence sample. Every displayed prediction is recalculated locally from the waveform. Browse the original millivolt view or the normalized view; filter N, V or mistakes; use previous/next buttons. No uploads, telemetry, storage, accounts or remote inference. Dynamic strings use textContent and a same-origin CSP limits resource loading. No browser runtime dependencies.

## Tests

Tests cover patient-group splits, preprocessing, PyTorch/export parity, exact confusion recomputation, checkpoint selection and all source hashes. JavaScript waveform/preprocessing/probability parity has a maximum error of 1.6e-7. Browser tests cover previous/next boundaries, label and mistake filters, waveform display, mobile overflow and runtime errors. Desktop and mobile renders were inspected.

`test_predictions.npz` holds every test label, probability and record for metric checks. These arrays and the downloaded data are local build artifacts, not required for the app. Re-running training creates the arrays needed by the Python metric test.

Browser harness: install Node.js and `npm install --no-save playwright`, then run `node browser-check.cjs`. Chrome defaults to `/usr/bin/google-chrome`; set `CHROME_PATH` to override. Screenshots save in `/tmp`; the test server binds to localhost.
