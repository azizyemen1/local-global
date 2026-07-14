# Combining Transformer Global and Task-Relevant Local CNN Features for Retail Product Recognition

Official implementation of the hybrid CNN-Transformer architecture for retail product recognition, combining Attentional Feature Fusion (AFF), Stages Feature Fusion (SFF), Transformer global representation, and attention-guided local feature refinement.

## Overview

Product recognition is a core task in retail computer vision systems. CNN-based methods capture local detail well but struggle with global correlations; Transformer-based methods capture long-range dependencies but overlook fine-grained local interactions and task-relevant focus.

This work proposes a hybrid architecture inspired by human visual perception — perceiving an object globally first, then attending to salient local details:

1. **AFF (Attentional Feature Fusion)** and **SFF (Stages Feature Fusion)** modules extract robust CNN features by aggregating and integrating information across multiple scales.
2. The fused CNN features are passed through a Transformer, whose self-attention mechanism models global correlations; multi-layer `[CLS]` tokens are integrated to form the final global representation.
3. Fused attention maps from multiple Transformer layers filter CNN local features to retain only task-relevant regions, suppressing background noise.
4. Concatenated global and refined local features are classified via **ArcFace**.

## Architecture

```
Input Image
   │
   ├─► CNN Backbone ─► AFF ─► SFF ─► Fused Local Features
   │                                       │
   │                                       ▼
   └─► Transformer Encoder ─► Multi-layer [CLS] fusion ─► Global Representation
                    │
                    ▼
        Multi-layer Attention Maps ─► Filter task-relevant local features
                    │
                    ▼
   Concatenate(Global Representation, Refined Local Features)
                    │
                    ▼
                 ArcFace
                    │
                    ▼
              Classification
```

## Datasets

| Dataset | Description |
|---|---|
| [Products-10K](https://products-10k.github.io/) | Large-scale retail product recognition benchmark |
| [RP2K](https://opendatalab.com/OpenDataLab/RP2K) | Retail product recognition, 2K+ categories |
| Retail1K | Commercial retail dataset (not publicly redistributable, see Data Availability) |

## Requirements

```bash
pip install torch torchvision timm einops
```
## Data Availability

Products-10K and RP2K are publicly available at the links above. The Retail1K dataset is not publicly available due to licensing restrictions imposed by the commercial provider from which it was acquired.
## Citation
## License

This repository is released for academic research purposes.
