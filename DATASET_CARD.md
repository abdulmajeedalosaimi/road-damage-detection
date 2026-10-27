# Dataset Card — RDD2022 Road-Damage Classification

## Intended use

This project classifies object crops produced from existing RDD2022 road-damage boxes.
It is a research prototype and must not be used alone for road-safety decisions.

## Audit

- Valid labelled images: 38384
- Valid D00/D10/D20/D40 objects: 55002
- Corrupt/unreadable images: 0
- Exact duplicate files removed: 1
- Visual-duplicate candidate groups: 2002

## Classes

| Class | Meaning            | Verified objects |
| ----- | ------------------ | ---------------: |
| D00   | Longitudinal crack |            26013 |
| D10   | Transverse crack   |            11829 |
| D20   | Alligator crack    |            10616 |
| D40   | Pothole            |             6544 |

## Leakage-safe split

Each collection-source group belongs to exactly one split.
China Drone and China MotorBike are kept together as the China source group.
This avoids training and testing on the same source, capture setup, and road environment.
It creates a realistic domain-shift test and is more reliable than an image-random split.
| Collection source | Split |
|---|---|
| China | val |
| Czech | val |
| India | train |
| Japan | train |
| Norway | test |
| United_States | train |

## Preprocessing

Each crop includes 10% context around a verified box
and is resized to 224x224.
Training augmentation is applied only to train crops; validation and test use deterministic transforms.

## Limitations

The classes are imbalanced; road conditions and camera viewpoints differ across countries.
Official RDD2022 test labels are withheld, so this project builds validation and test splits only from labelled training data.
RDD2022 provides boxes but no instance masks, so it is not ground truth for segmentation.

## Final M1 preparation and verification

### Dataset cleaning

- Training-folder images scanned: 38,385.
- Missing XML annotations: 0.
- Corrupt images detected: 0.
- Exact duplicate images excluded: 1.
- Images retained in the split manifest: 38,384.
- One confirmed incorrect D40 annotation was excluded from classification
  crop generation: Japan_Japan_003187, object index 0.
- Its coordinates match the original XML. The original XML and audit
  decision were preserved.

### Country-separated splits

- Training: India, Japan, United States.
- Validation: China and Czech.
- Test: Norway.
- China Drone and China MotorBike remain in the same source group.
- Validation and test use labeled training-folder data, not the official
  unlabeled test set.
- Whole-country separation takes priority over exact split percentages.
  Class-balance tables document the resulting imbalance.

### Duplicate screening

- No exact duplicate file content crosses splits.
- The coarse visual-hash check identified 2,002 candidate groups;
  388 contained images from different splits.
- These groups produced 22,735 cross-split candidate pairs.
- We visually reviewed the 30 highest-ranked pairs and 20 randomly
  selected remaining pairs. All 50 were judged to show different scenes.
- This limited screening does not prove that all near-duplicates are
  absent. Within-split candidates and pairs missed by the initial
  candidate search have not been comprehensively reviewed.

### Label-quality audit

### Visual Label Audit

The original audit used a reproducible random sample of **100 images**,
not a stratified sample of classification crops.

Each saved review decision was checked against the original sample
using both its audit ID and image ID. All records matched.

| Review outcome             |  Images |
| -------------------------- | ------: |
| Correct                    |      98 |
| Confirmed annotation error |       1 |
| Unresolved                 |       1 |
| **Total**                  | **100** |

- **Observed image-error fraction:** 1.00% (1/100).
- **Conservative fraction:** 2.00% (2/100), counting the unresolved
  image as a potential error.
- **Exact one-sided 95% Clopper–Pearson upper confidence bound:** 6.16%.

The audit unit is an image with at least one target annotation error,
not an individual crop or bounding box.

Audit 027 remains unresolved. Audit 044 contains a misplaced D40 box;
that annotation was excluded from classification crop generation,
while its original error status was preserved.

The confidence bound assumes the stated random-sampling and binomial
model and depends on the accuracy of the manual review. It does not
establish that unreviewed annotations are correct.

The later targeted image-quality review is separate. Two low-detail
validation crops remain marked REVIEW and are retained; they are not
added to the original audit counts.

### Automated crop and loader verification

- All 32,692 crops opened successfully as 224 x 224 RGB images.
- Crop labels matched source annotations and class folders.
- Saved JPG files matched the classification manifest.
- The excluded incorrect D40 crop was absent.
- All splits used D00=0, D10=1, D20=2, D40=3.
- One batch from each loader passed shape and finite-value checks.
- Loader smoke tests used num_workers=0; multiprocessing was not tested.
- Validation/test transforms were deterministic in the tested examples.

### Remaining limitations

- Small or blurry source regions can remain difficult to interpret
  after resizing. Resizing does not recover lost detail.
- Training random cropping can remove part of the damage; successful
  loading does not guarantee that every augmentation preserves it.
- Automated checks verify structure and inherited labels, not the
  visual correctness of every annotation.
- Country differences and class imbalance affect generalization.
- Test-accuracy confidence intervals have been reported in the
  Phase 1 report (Section 3.5) for the three evaluated classifiers.
