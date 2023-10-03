import matplotlib.pyplot as plt
from reading_dataset import *
import emd
from emd.sift import ensemble_sift





def denoising_data_emd (mat,sd_tresh=0.05 ):
  
    
    number_of_leads=mat.shape[0]
    number_of_steps=mat.shape[1]
    emd_imf_opts = {'sd_thresh': sd_tresh}
    
    emd_list=[]
    complete_emd_list=[]
    for i in range (6) :
        noisy_signal=mat[i]
        emd_imfs=emd.sift.sift(noisy_signal,imf_opts=emd_imf_opts )
        emd_sumation=np.sum(emd_imfs[:,:-3],axis=1 )

        complete_eemd_imfs=emd.sift.complete_ensemble_sift(noisy_signal,  nensembles=24, nprocesses=6, ensemble_noise=1, imf_opts=emd_imf_opts)
        complete_eemd_summation=np.sum(complete_eemd_imfs[0][:,:-3],axis=1)
        emd_list.append(emd_sumation)
        complete_emd_list.append(complete_eemd_summation)
    emd_mat= np.array(emd_list)
    complete_emd_mat= np.array(complete_emd_list)
    
    return (emd_mat, complete_emd_mat)
            





