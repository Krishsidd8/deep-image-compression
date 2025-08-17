# Optimal Image Compression via Lightweight Deep Learning Architectures

Official implementation of the paper:

**Optimal Image Compression through Integration of Deep Learning Architectures**  
Krish Siddhiwala, May 2025

[Project Page](#) | [Paper (PDF)](#) | [Dataset](https://www.kaggle.com/c/dogs-vs-cats)  

This repository contains code and training procedures for a lightweight deep learning-based image compression model combining Convolutional Autoencoders (CAEs), Residual Blocks (ResBlocks), and Generative Adversarial Networks (GANs). The model is designed for edge applications where memory and compute resources are constrained.

---

## 🔧 Key Features

- **High Compression Ratio:** Up to 48:1 with CAE-only setup
- **ResBlock Integration:** Enhanced detail preservation
- **GAN-augmented Training:** Improved perceptual quality with adversarial learning
- **Loss Balancing:** Adaptive weighting between L1, LPIPS, and GAN losses
- **Ablation Studies:** Quantitative and visual comparison of CAE, CRB, and GCRB variants

---

## 📷 Example Results

<p align="center">
  <img src="images/final_output.png" width="600"/>
</p>

Reconstructed images from different model configurations:  
**Left to Right**: Input → CAE → CAE+ResBlocks → CAE+ResBlocks+GAN

---

## 📁 Project Structure