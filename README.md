# Hybrid Deepfake Detection (CNN + ViT + Self-Distillation)

This repository implements the hybrid deep learning framework for DeepFake image detection as described in "A Hybrid Deep Learning Framework Integrating CNN and Vision Transformer with Self-Distillation for Robust DeepFake Image Detection".

**Features:**
- CNN for local feature extraction
- Vision Transformer (ViT) for global context
- Self-distillation for improved generalization
- Data augmentation and early stopping

## Usage

1. Place your dataset in the `data/real` and `data/fake` folders.
2. Install dependencies:
