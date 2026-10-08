# Music Classification through CNN and Classical Algorithms

## Overview

This project performs music genre classification using MFCC audio features, a two-layer Dilated Convolutional Neural Network (Dilated CNN), and classical machine learning algorithms.

The project investigates whether CNN-learned feature representations can improve classical classifiers compared with using original MFCC features.

An additional Layer 1 + Layer 2 feature-fusion experiment is also performed.

---

## Dataset

The project uses the GTZAN Music Genre Dataset.

Five genres are used:

- Blues
- Classical
- Hip-Hop
- Metal
- Pop

Dataset:

- 500 tracks
- 100 tracks per genre
- 400 training tracks
- 100 testing tracks
- Stratified 80:20 split
- Random state: 42

---

## Preprocessing

Audio is processed using LibROSA.

MFCC configuration:

- 20 MFCC coefficients
- 1293 time frames per track
- Dataset shape: `(500, 20, 1293)`

The MFCC features are standardized and reduced to 50 dimensions using PCA for the classical machine learning experiments.

PCA uses the full SVD solver for reproducibility.

---

## Dilated CNN

A two-layer 1D Dilated CNN is used.

### Layer 1

- Filters: 8
- Kernel size: 8
- Dilation rate: 4
- Average pooling: 32
- Dropout: 0.5
- Batch normalization

### Layer 2

- Filters: 16
- Kernel size: 4
- Dilation rate: 2
- Average pooling: 8
- Dropout: 0.5
- Batch normalization

### Classifier Head

```text
Flatten → Dense(128) → Softmax(5)