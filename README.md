# BSR-CLIP

PyTorch implementation of **BSR-CLIP: Background-Calibrated Structural Reasoning for  
Zero-Shot Visual-Language Anomaly Detection**.

<p align="center">
  <img src="./pic/ICMR-picture.pdf" alt="BSR-CLIP Framework" width="70%">
  <br>
  <em>Overview of BSR-CLIP. Zero-shot anomaly segmentation results are shown for cross-domain evaluation.</em>
</p>

---

## 📌 Overview

**BSR-CLIP** is a **cross-domain zero-shot visual-language anomaly detection** framework that improves anomaly detection robustness by jointly modeling **background-calibrated anomaly reliability** and **structure-aware spatial reasoning**.  
By leveraging pretrained CLIP representations, BSR-CLIP performs anomaly localization and detection without any target-domain training data.

This repository provides a complete PyTorch pipeline for training, evaluation, and ablation studies, enabling reproducible research and cross-domain benchmarking.


## 📂 Dataset Preparation

Please organize your dataset directory as follows. Ensure the folder structure strictly matches the layout below:

```text
data_dir/
├── Br35H
│   ├── no
│   └── yes
├── BrainMRI
│   ├── no
│   └── yes
├── btad
│   ├── 01
│   │   ├── ground_truth
│   │   │   └── ko
│   │   ├── test
│   │   │   ├── ko
│   │   │   └── ok
│   │   └── train
│   │       └── ok
│   ├── ...
├── CVC-ClinicDB
│   ├── images
│   └── masks
├── CVC-ColonDB
│   ├── images
│   └── masks
├── MPDD
│   ├── blacket_black
│   │   ├── ground_truth
│   │   │   └── hole
│   │   │   └── scratches
│   │   ├── test
│   │   │   ├── hole
│   │   │   └── good
│   │   │   └── scratches
│   │   └── train
│   │       └── good
│   ├── ...
├── ISIC2016
│   ├── ISBI2016_ISIC_Part1_Test_Data
│   └── ISBI2016_ISIC_Part1_Test_GroundTruth
├── Kvasir
│   ├── images
│   └── masks
├── mvtec 
│   ├── bottle
│   │   ├── ground_truth
│   │   ├── test
│   │   │   ├── broken_large
│   │   │   ├── broken_small
│   │   │   ├── contamination
│   │   │   └── good
│   │   └── train
│   │       └── good
│   ├── ...
├── visa
│   ├── candle
│   │   └── Data
│   │       ├── Images
│   │       │   ├── Anomaly
│   │       │   └── Normal
│   │       └── Masks
│   │           └── Anomaly
│   ├── ...
│   └── split_csv
