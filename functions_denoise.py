import numpy as np
from reading_dataset import *
from ecg2vcg import leads2vcg, limb2augmented
import statistics
import math
import scipy
from scipy.signal import butter, filtfilt
from denoise_wavelet import denoising_data
from denoise_emd import denoising_data_emd
import wfdb



def lowpass (signal):
    fs=500
    cutoff=10
    nyq=0.15*fs
    order=5
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, signal)
    return y
def lowpass_forall (mat):
    a=np.zeros(mat.shape)
    for i in range (len (mat)):
        denoised=lowpass(mat[i])
        a[i,:]=denoised
    return a  
    

def clean_the_ecg (mat,method='lowpass'):
    if method=='lowpass':
        clean_mat=lowpass_forall(mat)
        return (clean_mat)
    

def calculate_snr_percent(signal, noise):
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = np.sum(noise ** 2) / len(noise)
    snr = (signal_power / noise_power)
    return snr

def calculate_snr(signal, noise):
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = np.sum(noise ** 2) / len(noise)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_noise_signal(clean_signal, noise_signal, snr_db):
    # Calculate the power of the clean signal
    clean_power = np.sum(clean_signal ** 2) / len(clean_signal)

    # Calculate the power of the noise signal
    noise_power = np.sum(noise_signal ** 2) / len(noise_signal)

    # Calculate the power of the desired noise signal
    desired_noise_power = clean_power / (10 ** (snr_db / 10))

    # Calculate the coefficient
    coefficient = np.sqrt(desired_noise_power / noise_power)
    final_noise=coefficient*(noise_signal)
    
    return final_noise




def create_noise (mat , noise_method,snr_db=6 ,percent_of_file=1 ):
    
    if snr_db==None:
        return np.zeros(np.shape(mat))
    else: 
        mat=mat/1000

        length=len(mat[0])
        # noisy_mat=np.zeros((np.shape(mat)[0],np.shape(mat)[1]))
        a=np.shape(mat)
        noise_mat=np.zeros(a)
        thetas=[0,-np.pi/3,-np.pi*2/3 ,5*np.pi/6,np.pi/6,-np.pi/2]
        x=np.arange(0,length)
        if noise_method =='Artificial_bw_noise':
            a1=np.random.uniform(low=0.1,high=0.5)
            a2=np.random.uniform(low=0.1,high=0.3)
            b1=np.random.uniform(low=0.1,high=0.8)/1000
            b2=np.random.uniform(low=0.1,high=0.8)/1000

            a1=0.33
            a2=0.22
            # y= (a1*np.sin(b1*2*np.pi*x)+ a2*np.cos(b2*2*np.pi*x))*1000

            final_noise=(0.33*np.sin(0.0004*2*np.pi*x)+0.22*np.cos(0.0009*2*np.pi*x))
        elif noise_method=='wn'   :
            final_noise=0.05*np.random.randn(mat[0].size)
            final_noise=calculate_noise_signal(mat[0],final_noise,snr_db=snr_db)
        else:
            # rand=np.random.uniform(0,percent_of_file)
            # noisy_amount = 10000*percent_of_file
            # rand = (10000-noisy_amount)

            loc='dataset\\mit-bih-noise-stress-test-database-1.0.0\\'
                
            noise_file = loc +noise_method
            noises, fields = wfdb.rdsamp(noise_file, sampfrom=0 , sampto=5000)
            noise=noises[:,0]
            if snr_db!='whole':
                final_noise=calculate_noise_signal(mat[0],noise,snr_db=snr_db)
            else:
                final_noise=noise

        for i in range(len(thetas)) :
            noise_mat[i]=final_noise*np.cos(thetas[i])
            # plt.figure()
            # plt.plot(mat[i])
            # plt.plot(noisy_mat[i])

        return noise_mat  *1000 

          




def denoise_wavelet (noisy_mat ,wavelet_type,wavelet_denoise_treshold=0.13):
    if wavelet_type=='just_baseline':
        denoised_mat = denoising_data(noisy_mat,wavelet_type=wavelet_type,wavelet_denoise_treshold=wavelet_denoise_treshold,
        just_baseline=True)
    else :
        denoised_mat = denoising_data(noisy_mat,wavelet_type=wavelet_type,wavelet_denoise_treshold=wavelet_denoise_treshold,
        just_baseline=False)
    return denoised_mat   


# from skimage import restoration
# def denoise_wavelet2(mat , method ='BayesShrink', mode='soft' , wavelet_type='sym8'):
   
#     denoised_list=[]
#     for i in range (6) :
#         noisy_signal=mat[i]
#         denoised_signal_bayes=restoration.denoise_wavelet(noisy_signal, method =method, mode=mode , wavelet=wavelet_type,rescale_sigma='True')
#         denoised_list.append(denoised_signal_bayes)
#     denoised_mat= np.array(denoised_list)
#     return denoised_mat


def denoise_emd_baseline(noisy_mat, cemd=False ,sd_tresh=0.05):
    denoisedmat=denoising_data_emd (noisy_mat,sd_tresh=sd_tresh )
    if cemd== False :
        return denoisedmat[0]
    else:
        return denoisedmat[1]