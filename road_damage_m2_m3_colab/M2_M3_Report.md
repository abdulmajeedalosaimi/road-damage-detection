# M2 and M3 Road Damage Classification Report

**Project:** Road Surface Damage Detection from Vehicle-Mounted Cameras  
**Contribution:** Classification, transfer learning, evaluation and experiment tracking  
**Contributor account:** `alharby6626`  
**Experiment:** `simple_v1`  
**Date:** 24 October 2026

## 1 Purpose and completed work

Our part was to teach models to classify an existing crop of road damage into one of four categories. A crop is a small image cut from a larger road photograph. We used the prepared M1 crops and kept their original training, validation and test assignments.

We trained three models, compared them, selected the preferred model using validation results, and then tested the fixed models on images from Norway. We also examined mistakes, explained the limitations, proposed improvements to the training data, and saved the code, results and experiment records.

The selected model was a fully fine-tuned ResNet18 from **epoch 4**. It reached **85.65% validation accuracy** and **78.45% test accuracy**. Its macro F1 decreased from **0.78494 on validation** to **0.61935 on test**, showing that performance was weaker on the held-out country.

This report covers our M2 and M3 contribution. The initial M1 data preparation was existing team work. Detecting damage locations, segmenting damaged areas and building the final deployed system belong to the later project modules.

## 2 What the project PDF required

The project guidelines identify road damage classification as part of Project 12. Page 32 asks for classification of damage crops into longitudinal cracks, transverse cracks, alligator cracks and potholes. Pages 6 and 7 explain the required deliverables and assessment criteria [1].

| Requirement in the PDF | What it means in simple terms | What we completed |
|---|---|---|
| A classifier trained from scratch, M2 | Train a model that starts without previously learned image features | Trained a small CNN from random initialization |
| Learning curves and a best checkpoint, M2 | Show training progress and save the model version chosen by validation performance | Saved loss and accuracy curves, histories and the best checkpoint for every model |
| Error gallery, M2 | Show examples where the classifier was wrong | Displayed eight confident validation mistakes from the selected model |
| Accuracy, top-k accuracy and per-class metrics, M2 | Measure overall performance and performance for each damage category | Reported top-1 and top-2 accuracy, precision, recall, F1 and macro F1 |
| Confusion matrix and calibration, M2 | Show which categories are confused and whether confidence matches correctness | Produced confusion matrices, reliability diagrams and expected calibration error |
| Pretrained baseline and frozen versus full fine-tuning, M3 | Compare different ways of adapting a model that already learned from images | Trained frozen and fully fine-tuned ImageNet ResNet18 models |
| W&B tracking and comparison, M3 | Keep records that connect each experiment to its settings and results | Logged training, validation, test evaluation and comparison runs |
| Error analysis and a data-improvement plan, M3 | Use observed mistakes to suggest useful changes to training data | Wrote an analysis and a plan linked to specific validation examples |
| Requirements file and code repository, M3 | Record the software environment and keep the work available to the team | Saved the completed notebook and updated documentation and dependency files on the working branch |
| Reproducibility, Phase 1 | Use recorded settings and seeds so another member can repeat the work | Fixed the seed, recorded settings, preserved outputs and documented the procedure; independent teammate reproduction remains unverified |

The rubric assigns **20 points to classification across M2 and M3** and **8 points to reproducibility across Phase 1**. These are assessment categories, not a predicted score for this work. The classification criterion emphasizes correct comparison, per-class evaluation, calibration and a plan based on error analysis. It does not state a minimum accuracy target for full credit [1, p. 7].

We also reported a test-accuracy confidence interval, listed among the M1 metrics on page 6. It strengthens the evaluation of our classifier but does not replace the team's separate M1 audit requirements.

## 3 The data we used

The four categories were:

| Code | Category | Simple description |
|---|---|---|
| D00 | Longitudinal crack | A crack running roughly along the road direction |
| D10 | Transverse crack | A crack running roughly across the road direction |
| D20 | Alligator crack | A connected network of cracks |
| D40 | Pothole | A hole or depression in the road surface |

The existing classification dataset contained **32,692 crops**. This count is the prepared subset used in our experiments, not the size of the entire RDD2022 dataset.

| Split | Crops | Countries | Purpose |
|---|---:|---|---|
| Training | 12,000 | India, Japan, United States | Update the model weights |
| Validation | 9,463 | China, Czech | Compare models and select checkpoints |
| Test | 11,229 | Norway | Measure final performance on a held-out country |

Training contained 3,000 crops per category. The other splits kept their existing class distributions:

| Category | Training | Validation | Test |
|---|---:|---:|---:|
| D00 | 3,000 | 5,092 | 8,570 |
| D10 | 3,000 | 2,758 | 1,730 |
| D20 | 3,000 | 1,095 | 468 |
| D40 | 3,000 | 518 | 461 |
| **Total** | **12,000** | **9,463** | **11,229** |

The notebook checked required fields, valid categories, duplicate crop paths, separation of parent images across splits and the existence of crop files. It reused the supplied images rather than making a new crop dataset or changing the split.

Keeping Norway separate follows the country-holdout requirement. It tests how well the model handles a country that was not used for training or model selection [1, p. 32].

## 4 The three models and their settings

### Learning from scratch

The small convolutional neural network, or CNN, started from random weights. It used three convolution blocks to learn image features and a final layer with four outputs. It had to learn useful features directly from our road-damage training crops.

### Using a frozen pretrained model

The frozen model used ResNet18 weights previously trained on ImageNet. We kept its feature-extraction layers fixed and trained a new four-class output layer. The BatchNorm statistics in the frozen backbone were also kept fixed.

In simple terms, this approach reused existing image features and learned how to connect them to our damage categories.

### Fine-tuning the full pretrained model

The fine-tuned model also started from ImageNet ResNet18 weights, but all layers could change during training. This allowed it to adapt its image features to road damage.

The frozen and fully fine-tuned experiments each started independently from ImageNet weights. The fully fine-tuned model did not continue from the frozen model's training. The scratch model also had a different architecture, so the comparison does not isolate the effect of pretraining alone.

| Setting | Value |
|---|---|
| Runtime | Google Colab with an NVIDIA Tesla T4 |
| Random seed | 42 |
| Training length | 10 epochs per model |
| Batch size | 32 crops |
| Model input | 224 × 224 RGB pixels |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Scratch learning rate | 0.001 |
| Frozen and fine-tuned learning rate | 0.0003 |
| Loss function | Cross-entropy |
| Checkpoint selection | Highest validation macro F1 |

An **epoch** is one pass through the training data. A **checkpoint** is a saved copy of the model weights. The **learning rate** controls the size of the weight updates during training.

Training images were resized, randomly flipped horizontally and normalized using ImageNet statistics. Validation and test images were resized and normalized without random augmentation. We avoided random cropping because M1 had already cropped the damaged regions.

## 5 How we ran and evaluated the work

We followed this order:

1. Connected the notebook to Drive and loaded the prepared M1 data.
2. Checked the manifest and extracted the crop archive onto the Colab runtime for image loading.
3. Set the training configuration and fixed the random seeds.
4. Trained the scratch, frozen and fine-tuned models for ten epochs each.
5. Saved the checkpoint with the highest validation macro F1 for each model.
6. Compared the three saved checkpoints on the validation split.
7. Inspected validation mistakes and wrote the data-improvement plan.
8. Kept the model choice fixed and evaluated all three checkpoints on Norway.
9. Saved the test results and wrote their interpretation.

Validation acted as the development check. The test set acted as the final examination. We did not choose a model or adjust it after seeing its test results.

### What the metrics mean

| Metric | Simple meaning |
|---|---|
| Accuracy or top-1 accuracy | How often the model's first choice is correct |
| Top-2 accuracy | How often the correct answer appears in its two highest-scoring choices |
| Precision | For a category, how many predictions of that category are correct |
| Recall | For a category, how many actual examples the model finds correctly |
| F1 | A combined measure of precision and recall |
| Macro F1 | The average F1 across the four categories, giving each category equal weight |
| Loss | The error measured by the training objective; lower is better under the same evaluation setup |
| ECE | Expected calibration error, measuring the gap between confidence and observed accuracy across confidence bins |
| Confusion matrix | A table showing correct predictions and mistakes between categories |

We used macro F1 for selection because the validation and test categories have different sizes. Accuracy alone could hide weak performance on smaller categories.

For all reported results, predictions used the highest softmax probability, with no confidence-rejection threshold. Top-2 accuracy used the two highest scores. ECE used ten equal-width confidence bins. We did not apply temperature scaling or another confidence-calibration adjustment.

## 6 Training behavior and validation results

The learning curves showed different behavior. The scratch model improved on training data but had unstable validation results. The frozen model was more stable. Full fine-tuning improved training performance, while its validation loss rose later in training. This is a sign of overfitting: the model became better at the training examples without maintaining the same improvement on unseen examples.

For the fine-tuned model, the final epoch reached approximately **96.48% training accuracy**, but its final-epoch validation accuracy was approximately **79.38%**. We therefore used the saved best checkpoint, not the last epoch automatically.

The following table reports each model's best checkpoint on the same **9,463 validation crops**, using **224 × 224 inputs**. Accuracy columns are percentages; F1 and ECE use a 0–1 scale.

| Model | Best epoch | Accuracy | Top-2 accuracy | Macro F1 | ECE | Loss |
|---|---:|---:|---:|---:|---:|---:|
| Scratch | 8 | 73.87% | 88.18% | 0.63004 | 0.01737 | 0.73634 |
| Frozen | 10 | 82.31% | 93.51% | 0.72635 | **0.01014** | 0.50459 |
| Finetune | 4 | **85.65%** | **96.48%** | **0.78494** | 0.05760 | **0.45084** |

We selected **finetune at epoch 4** because it had the highest validation macro F1. The frozen model had the lowest validation ECE, meaning better calibration by this measure. Better calibration did not mean better classification performance: these metrics answer different questions.

## 7 Mistakes we examined and the improvement plan

The most frequent validation confusions for the selected model were:

| True category | Predicted category | Number of crops |
|---|---|---:|
| D00 | D20 | 428 |
| D00 | D10 | 230 |
| D10 | D20 | 121 |
| D20 | D40 | 85 |
| D40 | D00 | 84 |

These are raw counts. A larger category has more opportunities to produce errors, so these counts should be read with per-class precision, recall and F1.

We inspected eight validation mistakes with the highest confidence. Some crops were blurry or had low contrast. One visible example included prominent road markings. All eight displayed confidence values rounded to 1.00. Because we deliberately selected confident mistakes, the gallery does not show the typical confidence of every prediction.

| Observation | Proposed improvement to training data | How a future experiment would check it |
|---|---|---|
| Confusion between longitudinal and alligator cracks | Add varied examples of both categories, including road markings and textured backgrounds; review ambiguous training labels against original images | Compare D00 recall, D20 precision and the D00-to-D20 confusion rate |
| Blurry or low-contrast crops | Include examples with varied lighting and sharpness; test mild blur and contrast augmentation that preserves the damage category | Compare per-class F1 on the same validation split |
| Confident mistakes and rising validation loss during full fine-tuning | Add difficult crack and pothole examples from more locations | Compare validation macro F1 and ECE together |

These changes are **future proposals**. They were not implemented as part of the reported experiments. We did not change validation or test labels, and a wrong model prediction was not treated as proof that an annotation was wrong.

## 8 Final test results

After fixing the model choices, we evaluated all three best checkpoints on **11,229 Norway test crops**, using **224 × 224 inputs** and the same prediction rules used for validation.

| Model | Accuracy | Top-2 accuracy | Macro F1 | ECE | Loss |
|---|---:|---:|---:|---:|---:|
| Scratch | 54.55% | 86.89% | 0.42669 | 0.08334 | 0.96780 |
| Frozen | 57.32% | 90.28% | 0.48508 | 0.15102 | 1.02136 |
| Finetune | **78.45%** | **94.88%** | **0.61935** | **0.06534** | **0.61028** |

The fine-tuned model performed best among the three tested models. It remained the selected model because of the earlier validation decision, not because of these test results.

### Accuracy confidence intervals

We estimated uncertainty by resampling parent images 1,000 times, keeping crops from the same original image together.

| Model | Test accuracy | 95% confidence interval |
|---|---:|---|
| Scratch | 54.55% | 53.28% to 55.85% |
| Frozen | 57.32% | 56.10% to 58.63% |
| Finetune | 78.45% | 77.46% to 79.44% |

These intervals describe uncertainty from the sampled Norway images. They do not measure how much results would change across repeated training seeds or other countries.

### Results for each damage category

The selected fine-tuned model produced the following test results:

| Category | Precision | Recall | F1 | Test crops |
|---|---:|---:|---:|---:|
| D00 | 0.9779 | 0.7697 | 0.8614 | 8,570 |
| D10 | 0.8188 | 0.9012 | 0.8580 | 1,730 |
| D20 | 0.3127 | 0.5726 | 0.4045 | 468 |
| D40 | 0.2240 | 0.8373 | 0.3535 | 461 |

The model handled D00 and D10 better than D20 and D40. For potholes, recall was relatively high: it correctly recognized **386 of 461** real D40 crops. However, precision was low because many other crops were also predicted as potholes.

### Test confusion matrix

Rows show the true category and columns show the predicted category. Every entry is a crop count from the selected model's Norway test evaluation.

| True category | Predicted D00 | Predicted D10 | Predicted D20 | Predicted D40 |
|---|---:|---:|---:|---:|
| D00 | 6,596 | 281 | 526 | 1,167 |
| D10 | 65 | 1,559 | 55 | 51 |
| D20 | 66 | 15 | 268 | 119 |
| D40 | 18 | 49 | 8 | 386 |

The largest test confusion was **D00 predicted as D40**, with **1,167 crops**. This explains much of the low pothole precision. It is different from the largest validation confusion, which was D00 predicted as D20.

### What the test results tell us

D00 makes up **76.32% of the test crops**. For context, a rule that always guessed D00 would already reach 76.32% accuracy on this split. This is a calculated reference from the class counts, not an additional trained experiment. It shows why the model's 78.45% accuracy must be considered together with its per-class results and macro F1.

The decrease from validation macro F1 of 0.78494 to test macro F1 of 0.61935 shows a generalization gap. Different countries, image appearance, class distributions and crop detail may contribute, but these experiments do not establish one cause.

The test reliability diagram also shows overconfidence in higher-confidence bins: the model was correct less often than its confidence suggested. Its 94.88% top-2 accuracy does not mean 94.88% first-choice accuracy or guarantee reliable automatic use.

## 9 Problems encountered and how we handled them

| Problem or observation | What we did | Outcome |
|---|---|---|
| An older W&B installation conflicted with the Colab environment | Updated the setup to W&B 0.22.3 and protobuf 6.33.5, then restarted the runtime as needed | Imports, W&B login and experiment logging completed |
| The shared Drive folder name contained a trailing space | Corrected the path to match the actual folder name | The notebook found the manifests, crop archive and output location |
| The earlier workflow was difficult to manage | Used a simpler, self-contained notebook with visible code and separate training cells | The three experiments could be run and reviewed step by step |
| Validation performance did not keep improving with training | Saved each model's best validation-macro-F1 checkpoint | The selected fine-tuned checkpoint was epoch 4 rather than epoch 10 |
| Multiple notebook copies made it unclear which uploaded version contained the results | Consolidated the completed notebook in the intended work folder and checked its saved results | The designated notebook contains the completed results and interpretation |
| M2–M3 requirements had replaced the root M1 requirements | Restored the root file and kept the M2–M3 environment file inside our folder | The two contributions have separate environment records |
| Supporting documents still described the earlier implementation | Updated the README, verification notes, dependency files and notebook explanations | The documentation now identifies the completed notebook and its actual workflow |
| Documentation changes risked losing saved results | Edited Markdown cells in a separate notebook copy and compared the original code cells and outputs | All 26 code cells and all saved outputs, including six embedded images, were preserved |

The final documentation revision did not rerun training or evaluation. The working branch was kept separate; merging can be handled by the team when needed.

## 10 What we added and where to find it

Our work is stored on branch `feature/m2-m3-classification`, inside `road_damage_m2_m3_colab`.

| Deliverable | Contents |
|---|---|
| `Road_Damage_M2_M3_.ipynb` | Executable workflow, saved results, learning curves, error gallery, comparisons and written analysis |
| `README.md` | Explanation of the completed workflow, settings, results, output locations and reproduction steps |
| `VERIFICATION.md` | Evidence of execution, protocol checks and remaining verification limits |
| `requirements-colab.txt` | Pinned additional Colab setup dependencies |
| `requirements.txt` | Recorded environment versions, including the protobuf setup pin |
| `M2_M3_Report.md` | This explanation of the contribution and its results |
| Drive experiment folder | Best checkpoints, settings, histories, predictions and detailed evaluation reports |
| W&B project | Tracked runs, plots, per-class tables and model comparisons |

The `m2m3_*.py` files and `sync_notebook.py` are retained from an earlier implementation. The completed notebook is self-contained and does not use them to produce the reported results.

### Recorded environment

The main recorded versions and setup pin are:

```text
torch==2.11.0+cu130
torchvision==0.26.0+cu130
numpy==2.1.3
pandas==2.2.3
Pillow==11.3.0
matplotlib==3.10.0
scikit-learn==1.6.1
wandb==0.22.3
protobuf==6.33.5
```

The original notebook setup pins W&B and protobuf but does not pin scikit-learn in that installation command. A reproduction should match the recorded environment before training. The fixed seed improves repeatability but does not guarantee identical results on different hardware or software versions.

### Saved model location

The completed experiment was saved at:

```text
/content/drive/MyDrive/AI Project /Results/m2_m3/alharby6626/simple_v1
```

The `AI Project ` folder name ends with a space. The notebook paths can be changed for another account.

Each model has its own `best.pt`, `config.json`, `history.csv`, learning curves and validation/test reports. The selected checkpoint is `finetune/best.pt` inside the experiment folder. Large model checkpoints and datasets are kept outside GitHub.

### Experiment links

- [W&B project](https://wandb.ai/alharby6626-a/rdd2022-road-damage)
- W&B group: `alharby6626_simple_v1`
- [Validation comparison](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/21x5c025)
- [Test comparison](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/upe9cn2v)

| Model | Training run | Validation evaluation | Test evaluation |
|---|---|---|---|
| Scratch | [Training](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/9sfs09ni) | [Validation](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/pqy9o5pw) | [Test](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/1689trq0) |
| Frozen | [Training](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/3rofru5p) | [Validation](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/s1zx22dd) | [Test](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/x7knkc5b) |
| Finetune | [Training](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/40tjo6u4) | [Validation](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/c31hmrw9) | [Test](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/sm4nxanv) |

There are eleven recorded runs: three training runs, six evaluation runs and two comparisons. Evaluation runs include per-class tables, confusion matrices, reliability diagrams and learning curves.

## 11 Limits and handover status

The completed work provides a baseline comparison with a documented selection method and final country-holdout evaluation. Several limits remain relevant:

- **One training seed:** we have not measured variation across repeated training runs.
- **Weak performance on smaller categories:** D20 and D40 need improvement, even though the fine-tuned model performs best among the three experiments.
- **Country and class differences:** the test split has a different country and a much larger proportion of D00 crops.
- **Crop quality:** blur, contrast and resizing may remove useful detail, but individual failure causes were not proven.
- **M1 audit dependency:** existing visual-audit records need reconciliation by image identity before claiming a verified label-error percentage. Our structural manifest checks do not resolve this.
- **Handover checks:** access to W&B and Drive checkpoints from a teammate's account, and an independent reproduction, have not yet been confirmed.

The recorded data-improvement ideas are future experiments. Any new development should keep the existing results intact and avoid using the already observed test results to guide tuning while presenting that same test set as a fresh evaluation.

## 12 Sources and evidence

1. **AI251P Applied Computer Vision, Capstone Project: Guidelines and Example Topics**, October 2026, supplied as `Capstone_Project_Guidelines_UPDATED.pdf`: Table 5 on page 6 for module deliverables and metrics; Table 6 on page 7 for assessment; Project 12 on pages 32–33 for road-damage tasks and the country holdout.
2. **Completed notebook:** `Road_Damage_M2_M3_.ipynb`, experiment `simple_v1`, including its saved training logs, validation and test tables, plots and written interpretation.
3. **Experiment records:** the W&B runs and comparison links listed above.
4. **M1 structural checks:** `M1_validation_report.json`, recording crop counts, country assignments and manifest consistency checks.
5. **Documentation preservation check:** the revised notebook's 26 code cells, execution counts and all saved outputs were compared with the original executed notebook and found unchanged. The six embedded images were preserved exactly.
