# Evaluating Vision Transformers for Ordinal Diabetic Retinopathy Classification on Fundus Images

A comparative deep learning study benchmarking **EfficientNet** and **Swin Transformer-Tiny** for five-class diabetic retinopathy severity classification using retinal fundus images from the **APTOS 2019 dataset**. Models are evaluated using **Accuracy, Macro-F1, and Quadratic Weighted Kappa (QWK)**.

<p align="center">
  <img src="images/1.jpg" width="650">
</p>

---

## 1. Overview

This study investigates deep learning approaches for **five-class diabetic retinopathy (DR) severity classification** from retinal fundus images.

Two transfer-learning architectures were evaluated under a consistent experimental protocol:

- **TF-EfficientNet** — convolutional neural network baseline
- **Swin Transformer-Tiny** — hierarchical vision transformer using shifted-window self-attention

### Objective

The primary objective is to determine whether **transformer-based visual representation learning** provides improved performance over a convolutional baseline for retinal disease severity classification.

<p align="center">
  <img src="images/2.jpg" width="600">
</p>

<p align="center">
  <img src="images/3.jpg" width="600">
</p>

<p align="center">
  <img src="images/4.jpg" width="600">
</p>
