# Open-Set Livestock Identification using Energy-Based and Physics-Informed Deep Metric Learning

Biometric identification of cattle (and goats) from face and muzzle images that can **recognise registered animals and reject unregistered ones**.

An EfficientNetB0 backbone produces a 512-d embedding on the unit hypersphere. It is trained with ArcFace plus three additions:

- **Unknown rejection loss**, using synthetic unknowns generated on the fly.
- **Physics-informed constraints**, a bilateral symmetry loss and a Laplacian smoothness loss.
- **Energy margin loss**, a magnitude-scaled, asymmetric-temperature, adaptive-margin free-energy loss.





---

## Method

```
Input (224x224x3)
  -> EfficientNetB0 (last 30 layers fine-tuned, all BatchNorm frozen)
  -> GlobalAveragePooling -> Dropout(0.4)
  -> Dense(512, L2 reg 1e-4)  -> z~ (unnormalised)
                                  |-> Energy loss (z~, W)
  -> L2 normalise -> z (unit embedding)
                                  |-> ArcFace loss (z, W)
                                  |-> Unknown rejection loss (z, W)
                                  |-> Symmetry loss (z vs flip(x))
                                  |-> Laplacian loss (z vs x + noise)
```

Total objective:

```
L_total = L_arc + λ2·L_unknown + λ3·L_sym + λ4·L_energy + λ5·L_lap
```

| Loss | Purpose |
|------|---------|
| **ArcFace** (m = 0.5, s = 64) | Angular class separation on known animals |
| **Unknown rejection** | Pushes synthetic unknowns away from every class prototype |
| **Symmetry** | Image and its horizontal flip must give similar embeddings (stop-gradient on flipped branch) |
| **Laplacian smoothness** | Embedding must be stable under small Gaussian noise (stop-gradient on noisy branch) |
| **Energy margin** | Uses the pre-normalisation magnitude as a confidence signal. Magnitude-scaled energy, asymmetric temperatures (τ_known = 0.05, τ_unknown = 0.20), adaptive per-sample margin (δ0 = 0.2, γ = 0.1) |

**Synthetic unknowns:** known images are heavily augmented (brightness, contrast, saturation, hue, both flips, 90° rotation, Gaussian noise) to destroy identity cues, and assigned label `-1`.

**Curriculum (200 steps per epoch, batch size 32):**

| Phase | Epochs | LR | Known / Unknown per batch |
|-------|--------|----|---------------------------|
| 1 | 5  | 1e-3 | 32 / 0 |
| 2 | 15 | 1e-3 | 25 / 7 |
| 3 | 10 | 1e-4 | 19 / 13 |

Physics weights (λ3, λ4, λ5) are ramped linearly from 0 to their targets over the first 10 epochs.

**Inference:** gallery = per-identity mean embedding (re-normalised). Identity is the nearest centre by cosine similarity. A query is rejected as unknown if its maximum similarity is below a threshold τ\*.

## Ablation configurations

| Model | Loss | λ2 | λ3 = λ5 | λ4 |
|-------|------|----|---------|----|
| L1 | ArcFace | 0 | 0 | 0 |
| L2 | + Unknown rejection | 0.3 | 0 | 0 |
| L3 | + Symmetry + Laplacian | 0.3 | 0.01 | 0 |
| L4 | + Energy margin (full) | 0.3 | 0.01 | 0.05 |


Data, trained models and checkpoints are **not** stored in this repository .


```

A GPU is recommended for training. Evaluation can run on CPU.

## Dataset preparation

Datasets used: own cow face and muzzle datasets, Cattley, CFFD (Cow Frontal Face), BCM, Cattle Verification, CFID, a goat internal dataset and a goat external dataset (Mendeley).

1. Crop the face or muzzle region with OpenCV.
2. Arrange images in a flat per-identity layout:

```
dataset/
└── <split>/
    ├── cattle_0001/
    │   ├── img_01.jpg
    │   └── img_02.jpg
    └── cattle_0002/
```

3. Split **per identity** into train / validation / test before training, and keep the file lists fixed.
4. For open-set evaluation, hold out whole identities as unknowns. Never use them in training.









