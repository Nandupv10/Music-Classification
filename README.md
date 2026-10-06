# Music Classification through CNN and Classical Algorithms

## Problem Statement

Music genre classification is the task of automatically identifying the genre of a music track from its audio characteristics.

In this project, we investigate whether features learned by a Dilated Convolutional Neural Network (Dilated CNN) can improve the performance of classical machine learning algorithms for music genre classification.

The project compares classical algorithms using original MFCC features with classical algorithms using features extracted from two layers of a Dilated CNN.

## Dataset

The project uses the GTZAN Music Genre Dataset.

For this project, five music genres were selected:

- Blues
- Classical
- Hip-Hop
- Metal
- Pop

There are 100 audio tracks from each genre, giving a total of 500 tracks.

The dataset is divided into:
- Training set: 400 tracks
- Testing set: 100 tracks

## Preprocessing

The audio files are processed using LibROSA.

Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from each audio track to represent its acoustic characteristics.

The MFCC representation contains:
- 20 MFCC coefficients
- 1293 time frames

The resulting dataset has the shape:

`(500, 20, 1293)`

The data is split into training and testing sets using stratified sampling with an 80:20 split.

## Classical Machine Learning

Four classical machine learning algorithms are used:

- Logistic Regression
- Gaussian Discriminant Analysis (GDA)
- Random Forest
- Support Vector Machine (SVM)

These algorithms are first evaluated using the original MFCC features to establish baseline performance.

For GDA, Random Forest and SVM, the features are standardized and reduced to 50 dimensions using PCA.

## Dilated CNN

A two-layer 1D Dilated Convolutional Neural Network is trained using the MFCC representations.

The CNN uses:

- Dilated 1D Convolution
- Average Pooling
- Dropout
- Batch Normalization
- Fully Connected Layer
- Softmax Output Layer

The two convolution layers use different dilation rates to allow the network to capture patterns over a wider temporal region.

The standalone Dilated CNN achieves a test accuracy of 77%.

## CNN Feature Extraction

Instead of using the CNN only as a final classifier, features are extracted from the two convolutional layers.

These learned features are then provided to the four classical machine learning algorithms.

Two feature sets are evaluated:

- Dilated CNN Layer 1 features
- Dilated CNN Layer 2 features

## Results

### Original MFCC Features

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 72% |
| GDA | 81% |
| Random Forest | 79% |
| SVM | 84% |

### Dilated CNN Features

| Feature Set | Logistic Regression | GDA | Random Forest | SVM |
|---|---:|---:|---:|---:|
| Layer 1 | 84% | 83% | **89%** | 83% |
| Layer 2 | **88%** | 82% | 85% | 84% |

### Best Result

The best performance is obtained using:

**Dilated CNN Layer 1 Features + Random Forest = 89%**

This improves the Random Forest baseline from 79% using original MFCC features to 89% using Dilated CNN Layer 1 features.

## Conclusion

The results show that features learned by the Dilated CNN can improve the performance of classical machine learning algorithms.

Among the evaluated combinations, Random Forest using Layer 1 Dilated CNN features achieved the highest accuracy of 89%.

The results demonstrate that a CNN can be used not only as a standalone classifier, but also as a feature extractor whose learned representations can be combined with classical machine learning algorithms.

## Project Structure

```text
Music-Classification/
│
├── extract_mfcc.py
├── prepare_dataset.py
├── split_dataset.py
├── test_audio.py
│
├── train_logistic.py
├── train_gda.py
├── train_random_forest.py
├── train_svm.py
│
├── train_dilated_cnn.py
├── extract_dilated_cnn_layer1_layer2.py
│
├── train_dilated_cnn_layer1_logistic.py
├── train_dilated_cnn_layer1_gda.py
├── train_dilated_cnn_layer1_random_forest.py
├── train_dilated_cnn_layer1_svm.py
│
├── train_dilated_cnn_layer2_logistic.py
├── train_dilated_cnn_layer2_gda.py
├── train_dilated_cnn_layer2_random_forest.py
├── train_dilated_cnn_layer2_svm.py
│
└── README.md