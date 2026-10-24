# Verification — M2 and M3 Classification

**Review date:** 24 October 2026  
**Reviewed notebook:** `Road_Damage_M2_M3_.ipynb`  
**Experiment:** `simple_v1`  
**Random seed:** 42

## Scope

This document records the evidence available for the completed M2–M3 classification experiment. It distinguishes the completed Colab run from checks that still require independent confirmation.

The main notebook is self-contained. The accompanying `m2m3_*.py` files and `sync_notebook.py` belong to an earlier implementation and were not used to produce the reported notebook results.

## Notebook and execution evidence

The reviewed notebook contains:

- 46 cells, including 26 code cells.
- 16 code cells with saved outputs.
- Six embedded PNG images.
- No saved error outputs.
- Completed training logs for scratch, frozen and fully fine-tuned models.
- Training and validation learning curves for all three models.
- Validation and test comparison tables.
- A validation error gallery and written data-improvement plan.
- The selected model’s test per-class metrics, confusion matrix and reliability diagram.
- Written interpretation of the final test results.

The saved outputs contain links to eleven W&B runs: three training runs, six evaluation runs and two comparison runs.

These observations confirm that the reviewed notebook contains actual execution results. They do not establish that another environment will reproduce identical numerical results.

## Data and protocol checks

The notebook uses the existing M1 classification manifest and crop archive.

| Split | Crop count | Countries |
|---|---:|---|
| Training | 12,000 | India, Japan, United States |
| Validation | 9,463 | China, Czech |
| Test | 11,229 | Norway |

The notebook checks:

- Required manifest fields are present and non-null.
- Class labels belong to D00, D10, D20 or D40.
- Train, validation and test splits are present.
- Crop paths are unique.
- No parent image appears in multiple splits.
- Referenced crop files exist after extraction.

The earlier `M1_validation_report.json` records additional manifest consistency checks. These checks do not establish visual label correctness.

The implemented training and selection protocol uses:

- Seed 42 and 224 × 224 RGB inputs.
- Ten epochs and batch size 32 for each model.
- AdamW with weight decay 0.0001.
- Learning rate 0.001 for scratch and 0.0003 for both ResNet18 experiments.
- Horizontal flipping during training only.
- ImageNet initialization independently for frozen and fully fine-tuned ResNet18.
- A frozen backbone, including BatchNorm statistics, in the frozen experiment.
- Validation macro F1 for checkpoint and model selection.

The fine-tuned model was selected before final testing. Test results were used for reporting, without subsequent model tuning.

## Recorded evaluation results

| Model | Selected checkpoint epoch | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| Scratch | 8 | 0.63004 | 54.55% | 0.42669 |
| Frozen | 10 | 0.72635 | 57.32% | 0.48508 |
| Finetune | 4 | 0.78494 | 78.45% | 0.61935 |

For the selected fine-tuned model:

- Test top-2 accuracy: **94.88%**.
- Test ECE: **0.06534**.
- Test-accuracy 95% confidence interval: **77.46%–79.44%**.

The accuracy interval uses 1,000 bootstrap resamples of parent images, retaining their crops together.

Classification uses the highest softmax probability without a confidence-rejection threshold. ECE uses ten equal-width confidence bins, including predictions with confidence 1.0. Temperature scaling was not applied.

Evaluation exports and W&B runs include per-class precision, recall and F1, confusion matrices, reliability diagrams and aggregate metrics.

## Error analysis

The notebook documents:

- Validation confusion between D00 and D20.
- The largest test confusion, D00 → D40, involving 1,167 crops.
- Low test precision for D40 despite relatively high recall.
- Class imbalance, country differences and possible loss of image detail.
- Signs of overfitting during full fine-tuning.
- Proposed improvements to training examples, image-quality coverage and review of ambiguous training labels.

The eight displayed validation errors were selected for high confidence. Their confidence values round to 1.00; they are not a representative sample of all predictions.

The data-improvement plan describes future experiments. It does not claim that those improvements were implemented or that displayed labels were verified as incorrect.

## Traceability

- [W&B project](https://wandb.ai/alharby6626-a/rdd2022-road-damage)
- Group: `alharby6626_simple_v1`
- [Validation comparison](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/21x5c025)
- [Test comparison](https://wandb.ai/alharby6626-a/rdd2022-road-damage/runs/upe9cn2v)

The completed run used a Tesla T4 in Colab. The recorded PyTorch and Torchvision versions were `2.11.0+cu130` and `0.26.0+cu130`. The notebook setup pins `wandb==0.22.3` and `protobuf==6.33.5`.

The notebook saves model configurations, histories, best checkpoints and evaluation reports to Drive. Evaluation loads each model’s `best.pt`, rather than using its final training epoch automatically.

## Limits and handover checks

- An independent reproduction by another teammate has not yet been verified.
- Current reviewer access to W&B and the saved Drive checkpoints requires confirmation.
- Dependency files and setup instructions must match the completed notebook’s environment.
- Results come from one training seed. The bootstrap interval does not measure variation across training seeds or countries.
- M1 visual-audit records require reconciliation by image identity before a verified label-error percentage can be reported.
- The saved outputs demonstrate completion of this experiment; they do not establish deployment readiness.

The completed results should remain preserved. Any reproduction should use a separate experiment folder and follow the same fixed evaluation protocol.
