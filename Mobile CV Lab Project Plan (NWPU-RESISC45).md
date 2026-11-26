# Mobile CV Lab: Project Plan (NWPU-RESISC45)

Nov 23, 2026 · @Alper Karayaman

## Decisions at a glance

We classify aerial scene tiles from NWPU-RESISC45 (all 45 classes) with a fine-tuned MobileNetV2 baseline, test four improvement methods (one per member), and deploy the best compressed model on the course hardware.

| Item | Decision |
| --- | --- |
| Task | Land-use scene classification of aerial and satellite image tiles, 45 classes |
| Application scenario | Onboard scene tagging on a small Earth-observation satellite or high-altitude UAV: each tile is classified on the edge device so the relevant scenes are downlinked first |
| Dataset | NWPU-RESISC45, all 45 classes, 31,500 RGB tiles of 256 x 256 px |
| Baseline (Step 2) | MobileNetV2 (width 1.0, 224 x 224 input, ImageNet weights), fine-tuned in TensorFlow/Keras |
| Method 1 (Step 3) | Data augmentation: Denis |
| Method 2 (Step 3) | Hyperparameter tuning and regularization: Hasan |
| Method 3 (Step 3) | Quantization (INT8): Alp |
| Method 4 (Step 3) | Pruning: Kevin |
| Robustness / domain shift | Synthetic corruptions on the NWPU test set plus a cross-dataset test on UC Merced (19 shared classes) |
| Tooling | TensorFlow/Keras, TF Model Optimization Toolkit, TFLite export (same stack as the Phase 3 notebooks) |
| Compute | Dataset on the TUM NAS, training on the PC-pool GPUs, Colab only for quick tests |

The method owners are a proposal; swap among yourselves if someone prefers a different method. The fixed rule: every method is compared against the same frozen baseline, on the same split, with the same evaluation script.

## Task, scenario and dataset

NWPU-RESISC45 is the hardest of the three offered datasets, which gives our four methods room to show a measurable effect.

Why NWPU-RESISC45:

- 45 classes, 700 tiles each (perfectly balanced), ground resolution 0.2 to 30 m, collected from Google Earth over 100+ countries. Plain accuracy is a fair metric.
- EuroSAT (64 x 64 px, 10 classes) is close to saturated with pretrained CNNs, so improvements would be hard to show.
- About 426 MB in total: fits the 50 GB NAS easily and trains on one PC-pool GPU.
- 256 x 256 tiles match the 224 x 224 input of mobile ImageNet models with little resizing.
- A natural domain-shift partner exists: UC Merced (2,100 tiles, 21 classes, 0.3 m, USA). 19 of its 21 classes map 1:1 onto NWPU classes (all except agricultural and buildings), so we can test cross-dataset generalization without relabeling.

We use all 45 classes, not a 5-class subset: the results are more meaningful and the dataset is small enough.

**Split.** Use the standard split from Google Research's remote\_sensing\_representations (also used by torchgeo and by the Hugging Face copy `timm/resisc45`): 18,900 train / 6,300 validation / 6,300 test (60/20/20). All tuning happens on validation; the test set is only used for the final numbers of each experiment.

**Download.** Easiest is [timm/resisc45 on Hugging Face](https://huggingface.co/datasets/timm/resisc45) (splits included). The official link on the Moodle page works too. Copy the extracted images to the NAS in `train/`, `val/`, `test/` class folders.

**Mission requirements** (our own targets, used to judge the final recommendation in Step 6; adjust if you like):

| Requirement | Target |
| --- | --- |
| Model size | Course limit 20 MB; flight model 5 MB or less as INT8 |
| Top-1 accuracy on the NWPU test set | 90 % or more |
| Latency per tile on the provided hardware | Fixed once the Phase 4 hardware is known (for example 50 ms or less) |
| Robustness | Small accuracy drop under sensor noise, motion blur, haze, downlink JPEG compression and imagery from another region/sensor |

## Baseline model (Step 2)

The baseline is MobileNetV2 from `tf.keras.applications` with ImageNet weights (`include_top=False`), followed by GlobalAveragePooling, Dropout(0.2) and Dense(45, softmax).

Why MobileNetV2:

- Explicitly allowed by the course and the same family as the Phase 3 MobileNet notebook.
- Only standard operations (convolution, depthwise convolution, BatchNorm, ReLU6, add). These convert and quantize cleanly on practically every edge runtime. The deployment hardware is only revealed in Phase 4 (17 Dec), so the lowest-risk architecture matters.
- MobileNetV3 adds squeeze-and-excite and hard-swish layers. Google removed exactly these in EfficientNet-Lite because they are poorly supported on some accelerators and hurt post-training quantization.
- It fits the 20 MB limit with a large margin (2,257,984 backbone parameters + 57,645 head parameters):

```latex
\text{size}_{\text{FP32}} \approx 2.32 \times 10^{6} \times 4\ \text{B} \approx 9.3\ \text{MB}, \qquad \text{size}_{\text{INT8}} \approx 2.32 \times 10^{6} \times 1\ \text{B} \approx 2.3\ \text{MB}
```

**Fixed baseline recipe** (deliberately simple; every improvement is measured against it):

1. Resize 256 to 224 px; preprocess with `tf.keras.applications.mobilenet_v2.preprocess_input` (scales to \[-1, 1\]). The Phase 3 notebook rescales by 1/255, which does not match the ImageNet weights.
2. No data augmentation.
3. Phase A: backbone frozen, train the head for 5 epochs, Adam, learning rate 1e-3, batch 64.
4. Phase B: unfreeze the whole network, fine-tune for 15 epochs, Adam, learning rate 1e-4. Call the backbone with `training=False` so BatchNorm statistics stay frozen.
5. Early stopping on validation loss (patience 3), keep the best validation checkpoint, seed 42.
6. Save `baseline_v1.keras` (without optimizer state), its config file and one row in `results.csv`. From then on the baseline is frozen.

## Improvement methods (Step 3)

Two methods raise accuracy and robustness (M1, M2), two shrink and speed up the model for deployment (M3, M4); the final model combines the best of all four.

| Method | Owner | Targets | Variants to compare |
| --- | --- | --- | --- |
| M1 Data augmentation | Denis | Accuracy, robustness; no inference cost | Geometric only / + photometric / + MixUp or CutMix |
| M2 Hyperparameter tuning and regularization | Hasan | Accuracy | Learning-rate schedule, optimizer, regularization, input resolution |
| M3 Quantization | Alper | Size, latency | FP32 / float16 / dynamic-range INT8 / full INT8 / QAT |
| M4 Pruning | Kevin | Size, latency | 30 / 50 / 70 / 80 % sparsity, standard vs XNNPACK policy |

Every owner delivers the same four things: the method as an option in the shared code, a small sweep (3 to 5 runs, more for M2), the best configuration with its rows in `results.csv`, and 1 to 2 poster figures with 3 to 5 bullet points.

**M1 Data augmentation (Denis)**

- Geometric: horizontal and vertical flips, random 90 degree rotations, random zoom/crop. Aerial tiles have no "up", so rotations keep the label.
- Photometric: brightness, contrast, saturation and hue jitter.
- Mixing: MixUp or CutMix as a third variant.
- Keras preprocessing layers (RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness) in the training pipeline only.
- Report clean test accuracy and robustness-suite accuracy per variant: augmentation is the method most likely to improve robustness.

**M2 Hyperparameter tuning and regularization (Hasan)**

- Search on the validation set: learning rate and schedule (constant vs cosine decay with warm-up), optimizer (Adam vs AdamW vs SGD with momentum), weight decay, label smoothing (0 / 0.1), dropout (0.2 / 0.5), number of unfrozen layers, batch size.
- Input resolution sweep (160 / 192 / 224 px): it trades accuracy for latency and feeds the final recommendation.
- KerasTuner (Hyperband or random search) or a small manual grid; about 15 to 25 short runs, full-length training only for the best configuration.
- Report the best configuration, the accuracy gain and one sensitivity plot (for example accuracy vs learning rate).

**M3 Quantization (Alp)**

- TFLite converter variants: FP32 reference, float16, dynamic-range INT8, full-integer INT8 with a representative dataset of about 300 training images.
- If full INT8 loses more than about 1 percentage point: quantization-aware training (`tfmot.quantization.keras.quantize_model`), a few fine-tuning epochs.
- Report size (MB), accuracy and CPU latency per variant.
- Owns the export path to the hardware (Step 5). If the Phase 4 hardware uses its own toolchain, the same PTQ vs QAT comparison is repeated there.

**M4 Pruning (Kevin)**

- TF-MOT `prune_low_magnitude` with a polynomial-decay schedule to 30 / 50 / 70 / 80 % sparsity, fine-tune, `strip_pruning`, convert to TFLite.
- Report accuracy vs sparsity, gzipped size and latency.
- Plain unstructured sparsity mostly shrinks the compressed file and does not speed up inference. For real CPU speed-ups, also try the `PruneForLatencyOnXNNPack` policy, which prunes only 1x1 convolutions so TFLite can run them sparse ([TF-MOT guide](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_for_on_device_inference); MobileNetV2 needs a small padding change on its first convolution, shown there).
- At the end, quantize the pruned model together with Alper (pruned + INT8).

**Combined final model.** Best M1 + best M2 settings, retrained once by Hasan (final FP32 model). Kevin prunes it, Alper quantizes it, giving the deployment candidates: FP32, INT8, pruned + INT8.

**Optional stretch** (whoever finishes early): MobileNetV2 with width 0.5 / 0.75 and MobileNetV3-Small trained with the final recipe, as extra points on the accuracy vs latency plot.

## Shared setup, benchmark protocol and robustness experiment (Step 4)

One repository, one evaluation script and one `results.csv` are the single source of truth; every number on the poster comes from that file.

**Repository** (private, on LRZ GitLab or GitHub):

- `data/` split files and class list
- `src/data.py`, `src/train.py` (config-driven), `src/evaluate.py`, `src/robustness.py`, `src/export_tflite.py`, `src/benchmark_latency.py`
- `configs/` one YAML file per experiment
- `results/results.csv`, `figures/`

**Rules:** one experiment = one config file + one appended row (never overwrite rows); same split, same seed, same epoch budget for comparisons; test-set numbers only for finished configurations.

**Columns of `results.csv`:** exp\_id, owner, method, config, top1, top5, macro\_f1, params, size\_keras\_mb, size\_tflite\_mb, latency\_pc\_ms, latency\_device\_ms, acc\_corruptions\_mean, acc\_ucmerced, notes, date.

**Latency protocol** (identical for everyone): TFLite model, batch 1, 224 x 224 input, same PC-pool machine type, 1 CPU thread, 50 warm-up runs, then median of 200 timed runs. Repeated on the deployment hardware in Step 5.

**Robustness and domain-shift suite** (Denis builds it, everyone runs it on their best model through the same script):

| Test | What it simulates | Settings |
| --- | --- | --- |
| Gaussian noise | Sensor noise | 3 severities |
| Motion blur | Platform motion, jitter | 3 severities |
| Defocus blur | Optics out of focus | 3 severities |
| JPEG compression | Lossy downlink | Quality 50 / 30 / 10 |
| Brightness and contrast shift | Illumination, sun angle | 3 severities |
| Haze | Atmosphere, thin cloud | 3 severities |
| Down- then upsampling | Higher altitude, coarser ground resolution | Factor 2 / 3 / 4 |
| UC Merced, 19 shared classes | Different region and sensor (real domain shift) | No retraining; 45-way prediction counted correct if it equals the mapped NWPU class |

Report per model: accuracy per corruption and severity, mean corruption accuracy, drop vs clean accuracy, UC Merced accuracy, and the classes that drop most (input for the failure-case analysis).

## Tasks per workflow step

Each task has one owner and a due date; tick it off when done. Format: task · owner · due.

### Step 1: Choose task and dataset

- [x] Agree on this plan in the group chat · All · Tue 24 Nov
- [ ] Select NWPU-RESISC45 in the Moodle dataset poll (only one member per group) · Alp · Tue 24 Nov
- [ ] Email the instructor: which hardware and toolchain Phase 4 uses, so the export path is planned early · Alper · Tue 24 Nov
- [ ] Download NWPU-RESISC45, upload to the NAS, write split files and class list · Alp · Wed 25 Nov
- [ ] Data loader in `src/data.py` (preprocess\_input, caching, prefetch) · Alp · Thu 26 Nov
- [ ] Log in to a PC-pool GPU machine and run a test job that reads from the NAS · All · Fri 27 Nov
- [ ] Application scenario and mission requirements as poster text · Alp · Sun 29 Nov

### Step 2: Build baseline

- [ ] Repository, folder structure, `results.csv` template · Hasan · Wed 25 Nov
- [ ] `train.py`: config-driven, two-phase fine-tuning, early stopping, checkpoints, fixed seed · Hasan · Fri 27 Nov
- [ ] `evaluate.py`: top-1, top-5, macro-F1, per-class accuracy, confusion matrix, params, size; appends to `results.csv` · Kevin · Fri 27 Nov
- [ ] `export_tflite.py` (FP32 first, quantized variants added in M3) · Alper · Sun 29 Nov
- [ ] `benchmark_latency.py` following the latency protocol · Kevin · Sun 29 Nov
- [ ] Train, evaluate and freeze baseline v1 (weights, config, results row) · Hasan · Mon 30 Nov
- [ ] Baseline figures: training curves, confusion matrix · Kevin · Wed 2 Dec

### Step 3: Improve the model

- [ ] M1: implement the augmentation variants · Denis · Wed 2 Dec
- [ ] M2: set up the tuner or grid · Hasan · Wed 2 Dec
- [ ] M3: post-training quantization variants of baseline v1 (size, accuracy, latency) · Alper · Sun 6 Dec
- [ ] M4: sparsity sweep on baseline v1 · Kevin · Sun 6 Dec
- [ ] M1 and M2: first sweep finished · Denis, Hasan · Sun 6 Dec
- [ ] One slide of preliminary results per member · All · Tue 8 Dec
- [ ] Status deck for the feedback session (5 to 6 slides) · Alper · Wed 9 Dec
- [ ] Project Discussions and Feedback, 14:00; write down the feedback · All · Thu 10 Dec
- [ ] M3: quantization-aware training if full INT8 loses more than 1 percentage point · Alper · Sun 13 Dec
- [ ] M4: XNNPACK pruning-policy variant · Kevin · Sun 13 Dec
- [ ] Hand the best M1 configuration to Hasan · Denis · Mon 14 Dec
- [ ] Each method final: best configuration, results rows, 1 to 2 figures, 3 to 5 bullet points · All · Tue 15 Dec
- [ ] Train the combined final FP32 model (best M1 + best M2) · Hasan · Fri 18 Dec
- [ ] Prune the final model · Kevin · Tue 22 Dec
- [ ] Quantize the final and the pruned model into the deployment candidates (FP32, INT8, pruned + INT8) · Alper · Wed 23 Dec

### Step 4: Benchmark and analyze

- [ ] Robustness suite in `src/robustness.py`: corruptions + UC Merced class mapping · Denis · Wed 2 Dec
- [ ] Robustness results for baseline v1 · Denis · Thu 3 Dec
- [ ] Robustness results for each method's best model · each owner, collected by Denis · Tue 15 Dec
- [ ] Comparison table and accuracy vs PC-latency plot (bubble = size) for all variants · Kevin · Wed 23 Dec
- [ ] Robustness results for the deployment candidates · Denis · Thu 7 Jan
- [ ] Update table and plot with on-device latency · Kevin · Tue 12 Jan
- [ ] Failure-case analysis: confusion matrix of the final model, most confused class pairs, example misclassified tiles · Kevin · Tue 12 Jan

### Step 5: Deploy

- [ ] Read the Phase 4 material, set up the hardware toolchain, deploy baseline v1 as a smoke test · Alper · Wed 23 Dec (Thu 7 Jan lab session if there is no access before the break)
- [ ] Deploy the three candidates on the hardware · Alper · Sun 10 Jan
- [ ] On-device measurements: test-subset accuracy (must match the PC), median latency, throughput, memory, file size · Kevin with Alper · Tue 12 Jan
- [ ] Demo: folder of tiles in, predicted class and confidence on screen; photos and a screen capture for poster and video · Alper · Tue 12 Jan
- [ ] Bring hardware questions to the lab sessions on Thu 7, 14 and 21 Jan, 14:00 to 16:00 · Alper, Kevin

### Step 6: Final recommendation and deliverables

- [ ] Own poster section (text + figures) in `Poster Template.pptx` · All · Wed 6 Jan
- [ ] Assemble poster v1 (A0) · Denis · Sun 10 Jan
- [ ] Final recommendation: chosen model vs mission requirements, trade-offs, limitations · Alper · Wed 13 Jan
- [ ] Feedback on poster v1 at the lab session · All · Thu 14 Jan
- [ ] Poster v2 · Denis · Sun 17 Jan
- [ ] Video v1: 2 to 3 slides, MP4, no sound, 120 s max · Kevin · Sun 17 Jan
- [ ] Proofread; check every number on poster and video against `results.csv` · All (numbers: Kevin) · Tue 19 Jan
- [ ] Final check with the tutors at the lab session · All · Thu 21 Jan
- [ ] Export poster as A0 PDF + PPTX and video as MP4, upload to Moodle · Denis (poster), Kevin (video), Alper (upload) · Thu 21 Jan evening, one day before the Fri 22 Jan deadline
- [ ] Rehearse the poster pitch; everyone can explain all four methods · All · Wed 27 Jan
- [ ] Poster session in Ottobrunn, 14:00 to 16:00 · All · Fri 29 Jan

## Timeline

All method work closes on Tue 15 Dec, the deployment candidates are ready before the break on Wed 23 Dec, and we upload on Thu 21 Jan, one day before the deadline.

&#91;embedded content: project timeline · 23 Nov 2026 to 29 Jan 2027\]

After the break only two working weeks remain, with the lab sessions on Thu 7, 14 and 21 Jan as the contact points; that is why the poster sections are due on Wed 6 Jan.

## Who does what

Each member owns one method plus one shared piece of infrastructure and one deliverable, so the load is roughly even.

| Member | Method | Infrastructure | Steps 5 and 6 |
| --- | --- | --- | --- |
| Alper | M3 Quantization | Dataset, splits, data loader, TFLite export | Coordination and Moodle submissions, 10 Dec status deck, deployment lead, demo, final recommendation |
| Hasan | M2 Hyperparameter tuning and regularization | Repository, training script | Baseline v1, combined final FP32 model |
| Denis | M1 Data augmentation | Robustness and domain-shift suite | Robustness results, poster assembly (v1, v2, final) |
| Kevin | M4 Pruning | Evaluation and latency scripts | Results table, accuracy vs latency plot, failure cases, on-device measurements, video |

## Risks and open questions

The biggest risk is the unknown deployment hardware; the plan keeps every model exportable until Phase 4 says otherwise.

| Risk | Fallback |
| --- | --- |
| Hardware and toolchain unknown until 17 Dec (the Phase 3 notebook links the Xilinx Vitis AI workflow, so it may not be plain TFLite) | Standard-op MobileNetV2; ask the instructor now; keep `export_tflite.py` modular so a second exporter can be added |
| Phase 3 closure on 17 Dec is tentative | If it moves, shift only the Step 5 tasks; Step 3 deadlines stay |
| Not enough GPU time (Colab limits, shared PC pool) | Short schedules for sweeps, full training only for the best configurations; start long runs early in the week |
| Only two working weeks between the Christmas break (24 Dec to 6 Jan) and the deadline | Candidates ready by 23 Dec; poster sections drafted by 6 Jan |
| Combined model is worse than the best single method | Report it as a finding and recommend the best point on the accuracy vs latency plot |
| Numbers on the poster do not match the experiments | All numbers come from `results.csv`; Kevin checks on 19 Jan |

Open questions for the instructor: Which hardware and toolchain does Phase 4 use? Is the 20 MB limit for the file on the device (INT8) or for the trained model?
