# Crop and data-loader verification

All 32,692 crops passed readability, RGB mode, and 224 x 224 size checks.

- Source annotation classes match crop manifest labels.
- Class folders match manifest labels: zero mismatches.
- Every crop retains its parent image split; no parent crosses splits.
- On-disk JPG paths exactly match the classification manifest.
- The excluded Japan_Japan_003187 object 0 crop is absent.
- Train / validation / test counts: 12,000 / 9,463 / 11,229.
- All loaders use D00=0, D10=1, D20=2, D40=3.
- One batch from each split loaded as [32, 3, 224, 224], with finite values.
- Actual transform definitions and dataset wrapper were extracted from train_classifier.py without invoking training.
- Repeated evaluation transformation of one image per evaluation split was identical.

## Limitations

These checks validate structure and inherited labels, not the visual correctness of every annotation.
Small or blurred source boxes can lose useful detail when enlarged to 224 x 224. The manually inspected India_India_003334_0.jpg is an example requiring source-image review; it was not deleted.
Training RandomResizedCrop can cut off damage. A successful batch does not establish that all augmentations preserve class evidence.
Loader smoke tests used num_workers=0; the configured multiprocessing setting was not tested.
No model training was performed. The reports are saved in the chat workspace, not yet incorporated into the project dataset card.
