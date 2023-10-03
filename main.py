from functions_denoise import *
from functions_compare import *
from reading_dataset import *

def improve_limb_leads (denoised_limb_leads_mat):
    shape = denoised_limb_leads_mat.shape
    assert shape[0] ==6
    listing=ecg2twoD(denoised_limb_leads_mat)
    x1,y1=mean_denoise(listing)
    reconstructed_ecg=reproduce_leads_from_denoise(x1,y1,denoised_limb_leads_mat)
    return reconstructed_ecg

if __name__ == "__main__":
    ecg=read_ecg(1)
    mat=ecg.mat [0:6,:]
    plot_mat(improve_limb_leads(mat), just_limb=True)
    plt.show()