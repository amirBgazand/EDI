from functions_denoise import *
from functions_compare import *
from reading_dataset import *

#%% Reading Dataset
index= 1
ecg = read_ecg(index)
mat = ecg.mat[:6,:]


denoise_method='wl'
wavelet_type = 'sym8'

#%% add some noise

artificial_bw_noise = create_noise(mat , noise_method='Artificial_bw_noise',snr_db=6)
noisy_mat = mat+ artificial_bw_noise

#%% Denoise
# emd_denoised= denoise_emd_baseline(mat, cemd=False)
wavelet_denoised= denoise_wavelet(mat, wavelet_type='sym8', wavelet_denoise_treshold=0.13)


#%% Improve Denoised limb leads
from main import improve_limb_leads
improved_leads=improve_limb_leads(wavelet_denoised)

#%% plotting ECG

plot_ecg(ecg , just_limb=True , time=(0,8))
plot_mat(wavelet_denoised, just_limb=True,time=(0,8))
plot_mat(improved_leads, just_limb=True,time=(0,8))

plt.show()
