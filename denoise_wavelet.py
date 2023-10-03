import numpy as np
import pywt
import sys

def calc_baseline(signal):
    
    ssds = np.zeros((3))

    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        if iterations >150:
            return np.zeros(len(signal))
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return baseline[: len(signal)]

def denoising_data(a_person_data,wavelet_denoise_treshold,wavelet_type='sym8',just_baseline=False) :

    number_of_leads=a_person_data.shape[0]
    number_of_steps=a_person_data.shape[1]

    denoised_baseline_data=np.zeros((number_of_leads,number_of_steps))

    for i in range(number_of_leads) :
        denoised_data_array=np.zeros((number_of_steps))
        noisy_data=[]
        counter=0

        # step1-removing powerline interface for each lead
        for j in range(number_of_steps) :
            noisy_data.append(a_person_data[i,j])
            if a_person_data[i,j]==0 :
                counter=counter+1
        
        if counter==len(noisy_data) :
            for j in range(number_of_steps) :
                denoised_baseline_data[i,j]=noisy_data[j]
            continue
        
        if just_baseline ==True:

            for j in range(number_of_steps) :
                denoised_data_array[j]=float(noisy_data[j])

            datarec=denoised_data_array
        else:
            w=pywt.Wavelet(wavelet_type)
            maxlev = pywt.dwt_max_level(len(noisy_data), w.dec_len)
            threshold = wavelet_denoise_treshold
            coeffs = pywt.wavedec(noisy_data, wavelet_type, level=maxlev)


            for coeff_num in range(1, len(coeffs)):
                coeffs[coeff_num] = pywt.threshold(coeffs[coeff_num], threshold*max(coeffs[coeff_num]))

            datarec = pywt.waverec(coeffs, wavelet_type)


        # step2-removing baseline wander for each lead
        base_line=calc_baseline(datarec)
        ecg_out=datarec-base_line

        # making n*steps matrix
        for j in range(number_of_steps) :
            denoised_baseline_data[i,j]=ecg_out[j]


    return denoised_baseline_data




