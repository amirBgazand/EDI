from functions_denoise import *
from functions_compare import *
from reading_dataset import *
from main import improve_limb_leads
#%% Reading Dataset
''' 
    Step by step guideline
    You have the option to change the index or import your own ECG.
    If importing your own ECG, begin by converting it to a numpy array.
    Next, divide the six limb leads to create an array with 6*n values, encompassing
    the leads I, II, III, aVR, aVL, and aVF respectively within your array.
    
    '''

index=558
ecg = read_ecg(index)
mat = ecg.mat[:6,:]


#%% 1-Add some noise
artificial_bw_noise = create_noise(mat , noise_method='Artificial_bw_noise',snr_db=6) 
noisy_mat = mat+ artificial_bw_noise

#%% 2-Denoise
denoise_method='emd' # 'emd' or 'wl'
if denoise_method =='wl':
    primary_denoised= denoise_wavelet(noisy_mat, wavelet_type='sym8', wavelet_denoise_treshold=0.13)
elif denoise_method =='emd':    
    primary_denoised= denoise_emd_baseline(noisy_mat, cemd=False)

#%% 3- Reconstruct Improved Denoised limb leads
improved_leads=improve_limb_leads(primary_denoised)

#%% plotting ECG
plot_mat(mat, just_limb=True, time=(0,8), title='clean signals')
# plt.savefig(f'dataset\\{index} -1 mat_ecg ')
plot_mat(noisy_mat, just_limb=True , time= (0,8),title='noisy signals')
# plt.savefig(f'dataset\\{index} -2 noisy_ecg ')
plot_mat(primary_denoised, just_limb=True,time=(0,8),title='primary denoise signals')
# plt.savefig(f'dataset\\{index} -3 denoise ')
plot_mat(improved_leads, just_limb=True,time=(0,8),title ='improved denoise signals')
# plt.savefig(f'dataset\\{index} -4 improve ')
plt.show()