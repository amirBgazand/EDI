# Vector Based Post-Processing Method for Improving ECG Denoising Techniques by Re-establishing lead Relationships

## Overview

This repository contains the implementation of a vector-based post-processing method designed to improve ECG denoising techniques by re-establishing lead relationships. The method outlined here is based on the research presented in the article titled ["Vector Based Post-Processing Method for Improving ECG Denoising Techniques by Re-establishing lead Relationships"](https://ieeexplore.ieee.org/abstract/document/10330088/authors#authors).


![Illustration of the procedure](images/overview.png)

*Figure 1: Step A) Denoising noisy ECG limb leads by an arbitrary system. Step B) Calculating 15 cardiac vectors from the denoised signals, then deriving primary average points for limb leads. These primary averages determine weights for the six limb leads, resulting in the final weighted average cardiac vector. Step C) By reconstructing lead values over time, our method effectively reduces limb lead noise.*


## Abstract
Reducing noise in electrocardiogram (ECG) signals is essential for accurate diagnosis and monitoring of cardiac diseases. Existing denoising methods often denoise leads individually and distort Einthoven’s law due to their limitations in distinguishing between noise and signal components across all leads. In this paper, a new method has been proposed that not only restores the Einthoven relationship but also effectively reduces the remaining noises of the denoised signals. We introduced the Weighted Average Cardiac Vector, calculated from denoised signals, and used it to reconstruct lead signals. Our method is applicable to all denoising processes to improve the denoised limb lead values and reduce the error signals. In experiments involving empirical mode decomposition (EMD) and wavelet transform, our post-processing achieved an average noise reduction of 28.6% and 10.8%, respectively. To assess our method in practice, we build two CNN-based diagnosis detection models (standard and ResNet-18). When our method was used alongside the wavelet transform, the standard CNN’s accuracy improved from 63% to 70%, while the ResNet’s accuracy increased from 73.31% to 77.41%. In the case of EMD, our method enhanced the standard CNN’s accuracy from 79.8% to 84.5% and the ResNet’s accuracy from 89.78% to 93.84%. These promising results recommend our algorithm as a valuable post-processing unit for denoising systems.



## Data Folder


To use the dataset used in this project, follow these steps:

1. Download the dataset from the [PhysioNet Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/) website.

2. The G12EC dataset can be found in the `training/georgia` directory on the provided link.

3. After downloading the dataset, make sure to organize it as follows:

    ```
    ├── dataset
    │   └── 6-PhysioNetChallenge2020_Training_E
    │       ├── All_HEA_Files_here
    │       └── All_MAT_Files_here
    ```

   You should place all the `.hea` and `.mat` files in their respective folders within the `6-PhysioNetChallenge2020_Training_E` directory.

Now your data is ready to be used for the project.

