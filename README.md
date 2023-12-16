# Vector Based Post-Processing Method for Improving ECG Denoising Techniques by Re-establishing lead Relationships

<br>

## Overview

This repository contains the implementation of a vector-based post-processing method designed to improve ECG denoising techniques by re-establishing lead relationships. The method outlined here is based on the research presented in the article titled ["Vector Based Post-Processing Method for Improving ECG Denoising Techniques by Re-establishing lead Relationships"](https://ieeexplore.ieee.org/abstract/document/10330088/).
<br>
<br>



## Abstract
Reducing noise in electrocardiogram (ECG) signals is essential for accurate diagnosis and monitoring of cardiac diseases. Existing denoising methods often denoise leads individually and distort Einthoven’s law due to their limitations in distinguishing between noise and signal components across all leads. In this paper, a new method has been proposed that not only restores the Einthoven relationship but also effectively reduces the remaining noises of the denoised signals. We introduced the Weighted Average Cardiac Vector, calculated from denoised signals, and used it to reconstruct lead signals. Our method is applicable to all denoising processes to improve the denoised limb lead values and reduce the error signals. In experiments involving empirical mode decomposition (EMD) and wavelet transform, our post-processing achieved an average noise reduction of 28.6% and 10.8%, respectively. To assess our method in practice, we build two CNN-based diagnosis detection models (standard and ResNet-18). When our method was used alongside the wavelet transform, the standard CNN’s accuracy improved from 63% to 70%, while the ResNet’s accuracy increased from 73.31% to 77.41%. In the case of EMD, our method enhanced the standard CNN’s accuracy from 79.8% to 84.5% and the ResNet’s accuracy from 89.78% to 93.84%. These promising results recommend our algorithm as a valuable post-processing unit for denoising systems.
<br>
<br>


## What we have done in a nutshell

There is a rule in ECG limb lead signals named Einthoven's law. This law clarifies that :

$$
\begin{flalign}
& \text{lead I} + \text{lead III} = \text{lead II} &
\end{flalign}
$$

and

$$ 
\begin{flalign}    
& \text{lead avR} + \text{lead aVL} + \text{lead aVF} = 0 &
\end{flalign}
$$ 

We discovered this law is only established before denoising techniques. However, the uniformity property and Einthoven's law are invalid when we use denoising algorithms. We have shown this in the article. Denoising algorithms can mistakenly remove crucial heart signals while leaving some noise due to their inability to precisely distinguish noise from true signals. 
Here, we utilize the discrepancy of the denoised curves to obtain a single curve and reproduce all leads, in which not only the mathematical relationship between leads is re-established, but also the noise of the leads is significantly reduced after any arbitrary denoising process. Table I summarizes the relationships between the frontal leads before and after denoising and after applying our post-processing method.

<be>

The proposed method aims to enhance denoising systems based on cardiac vectors derived from pairs of leads in ECG signals. The process involves three main steps:

1. **Step A - Denoising:**
   - Utilizes any arbitrary denoising technique (like wavelet transform or EMD) to remove noise from the six limb leads, producing denoised frontal limb lead values.

2. **Step B - Cardiac Vector Calculation:**
   - Constructs 15 cardiac vectors from the denoised lead values.
   - Aligns these vectors and calculates primary average points for each lead pair.
   - Determines weights for the leads based on proximity and variance, resulting in a fianl WEIGHTED AVERAGE CARDIAC VECTOR.

3. **Step C - Reconstruction:**
   - Projects the improved cardiac vector onto the frontal plane.
   - Derives denoised limb lead values by projecting this vector onto the limb lead direction over time.
     
This method intends to enhance denoising accuracy by aligning and aggregating information from multiple leads, ultimately improving the quality of ECG signals by minimizing noise across limb leads.

read the full article ["here"](https://ieeexplore.ieee.org/abstract/document/10330088/). 

<p align="center">
  <img src="images/overview.png" alt="Illustration of the procedure" width="800">
</p>

<br>

***
<div align="center">
  <strong>WHAT IS YOUR DENOISING TECHNIQUE? IT DOES NOT MATTER; USE OUR POSTPROCESSING ALGORITHM TO IMPROVE IT. </strong>
</div>
      
***


<br>
<br>

## Example
Here is an example of our improvement algorithm. In this case, artificial bw noise was added to a clean ECG and then denoised using an EMD denoising process. Although the denoised signals (Fig. 12c) still contained some errors compared to the clean signals, our algorithm significantly reduced these errors in leads I, -aVR, and lead III. As a result, the RMSE was reduced by 52%.

<p align="center">
  <img src="images/result1.png" alt="An example of our improvement algorithm" width="600">
</p>

      
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
<br>

## Citation

If you find this work helpful, please consider citing:

A. Ghafari, N. Pourjafari and A. Ghaffari, "Vector Based Post-Processing Method for Improving ECG Denoising Techniques by Re-establishing lead Relationships," in *IEEE Transactions on Instrumentation and Measurement*, doi: [10.1109/TIM.2023.3335528](https://ieeexplore.ieee.org/document/10330088)
