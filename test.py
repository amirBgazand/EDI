from functions_denoise import *
from functions_compare import *
from reading_dataset import *
from main import improve_limb_leads
#%% Reading Dataset
'''you can change the index or just import your own ECG
    if you import your own ECG you have to convert it to numpy array first
    then you have to devide six limb leads to have 6*n array including leads
    I, II, III, aVR, aVL, aVF respectively in the your array
    '''

index=558
ecg = read_ecg(index)
mat = ecg.mat[:6,:]


#%% add some noise
artificial_bw_noise = create_noise(mat , noise_method='Artificial_bw_noise',snr_db=6) 
noisy_mat = mat+ artificial_bw_noise

#%% Denoise
denoise_method='emd' # 'emd' or 'wl'
if denoise_method =='wl':
    primary_denoised= denoise_wavelet(noisy_mat, wavelet_type='sym8', wavelet_denoise_treshold=0.13)
elif denoise_method =='emd':    
    primary_denoised= denoise_emd_baseline(noisy_mat, cemd=False)

#%% Improve Denoised limb leads
improved_leads=improve_limb_leads(primary_denoised)

#%% plotting ECG
plot_mat(mat, just_limb=True, time=(0,8))
plt.savefig(f'dataset\\{index} -1 mat_ecg ')
plot_mat(noisy_mat, just_limb=True , time= (0,8))
plt.savefig(f'dataset\\{index} -2 noisy_ecg ')
plot_mat(primary_denoised, just_limb=True,time=(0,8))
plt.savefig(f'dataset\\{index} -3 denoise ')
plot_mat(improved_leads, just_limb=True,time=(0,8))
plt.savefig(f'dataset\\{index} -4 improve ')