# Additional image-quality review

## Scope
Two flagged China Drone images were inspected using their original
bounding-box regions without square resizing. Both contribute one
crop to the validation set.

## Findings
- **China_Drone_China_Drone_000581 (D10):** The original box region is
  95 × 26 pixels. A faint horizontal crack-like feature is visible,
  but limited detail prevents confident label verification.
- **China_Drone_China_Drone_000997 (D00):** A narrow vertical crack-like
  feature is visible through portions of painted markings, but the
  label cannot be confidently verified.

## Decision
Both records remain REVIEW and their crops are retained.
These are uncertain cases, not confirmed annotation errors.
No images, crops, labels, or split assignments were changed.

## Dataset limitation
Some source annotations cover narrow or low-detail regions. Resizing
these regions directly to 224 × 224 distorts their proportions and
can make visual interpretation harder. Inspection without resizing
does not resolve all label uncertainty.

## Relationship to the label audit
This targeted quality review is separate from the random label audit.
Its findings must not be added to the random audit counts unless the
images are verified members of that audit sample.
