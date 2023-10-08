import matplotlib.pyplot as plt
from functions_compare import ecg2twoD , mean_denoise, reproduce_leads_from_denoise
from reading_dataset import read_ecg , plot_mat

def improve_limb_leads(denoised_limb_leads_mat):
    assert denoised_limb_leads_mat.shape[0] == 6
    listing = ecg2twoD(denoised_limb_leads_mat)
    x1, y1 = mean_denoise(listing)
    reconstructed_ecg = reproduce_leads_from_denoise(x1, y1, denoised_limb_leads_mat)
    return reconstructed_ecg

if __name__ == "__main__":
    ecg = read_ecg(1)
    mat = ecg.mat[0:6, :]
    improved_ecg = improve_limb_leads(mat)
    plot_mat(improved_ecg, just_limb=True)
    plt.show()