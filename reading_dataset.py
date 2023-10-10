import pickle
import regex as re
import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import pickle as pkl


class data ():
    """Class to represent ECG data."""
    def __init__(self,index,mat,age, sex,Dx,samples
    ,sample_rate,Rx=None,Hx=None,Sx=None,folder=None) :

        self.folder=folder
        self.index=index
        self.age=age
        self.sex=sex
        self.dx=Dx
        self.rx=Rx
        self.hx=Hx
        self.samples=samples
        self.sample_rate=sample_rate
        self.mat=mat

    def find_duration(self):
        if self.samples != None:
            self.duration=self.samples /self.sample_rate
        return self.duration 

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def get_location(folder_number):
    folders=['1-Training_WFDB','2-Training_2','3-PhysioNetChallenge2020_Training_StPetersburg'
        ,'4-PhysioNetChallenge2020_Training_PTB','5- PhysioNetChallenge2020_Training_PTB-XL','6-PhysioNetChallenge2020_Training_E']
    location= 'dataset\\'+folders[folder_number]  +'\\'
    pishvands=['A','Q','I','S','HR','E']
    pishvand=pishvands[folder_number]
    zfilling_number= [5 if folder_number==5 or folder_number==4 else 4][0] 
    return location , pishvand , zfilling_number
   

def get_ecg(index,folder_number):
    location , pishvand , zfilling_number=get_location(folder_number)
    ecg_annotation_file=f'{location}{pishvand}{(str(index)).zfill(zfilling_number)}.hea'
    ecg_mat_file=f'{location}{pishvand}{(str(index)).zfill(zfilling_number)}.mat'
    return ecg_mat_file, ecg_annotation_file


def read_ecg(index,folder_number=5 ,):
    """Read ECG data from files and create an ECGData object."""
    ecg_mat_file,ecg_annotation_file=get_ecg(index,folder_number)
    mat = scipy.io.loadmat(ecg_mat_file)
    np.seterr(invalid='ignore')
    mat=mat['val']

    with open (ecg_annotation_file, 'r') as file:
        
        lines=file.readlines()
        first_line=lines[0]
        first_line=first_line.split(' ')
        ecg_sample_rate=float(first_line[2])
        ecg_samples=float(first_line[3])
        
        age_line=lines[13]
        ecg_age=re.findall(r'Age:\s(\d*)',age_line)[0]
        sex_line=lines[14]
        ecg_sex=re.findall(r'Sex:\s(.*)',sex_line)[0]
        dx_line=lines[15]
        ecg_dx=re.findall(r'\d+',dx_line)

    ecg=data (folder=folder_number ,index=index,mat=mat, age=ecg_age, sex=ecg_sex,
              Dx=ecg_dx, sample_rate=ecg_sample_rate, samples=ecg_samples,)  
    ecg_duration=ecg.find_duration()  

    return ecg


def dx2diagnosis(dx):
    diagnosis_dict=pkl.load(open('dataset\\dx.pkl' , 'rb'))
    return (diagnosis_dict[dx[0]])


def plot_ecg (ecg,time=None,channels=None ,folder_number=5 
              , custom_order=True , just_limb=False,subtitle=True):
    
    mat=ecg.mat
       
    if time != None:
        assert type(time)==tuple  
    if channels != None:
        subplot(channels,ecg,mat)
    else:
        plot_mat(mat,time ,custom_order , just_limb=just_limb)
        if subtitle ==True:
            diagnosis=dx2diagnosis(ecg.dx)
            plt.suptitle(f'index:{ecg.index} , age:{ecg.age} , sex:{ecg.sex} , dx:{str(diagnosis)} '
                          , fontsize=9 )


def plot_mat (mat, time=None, custom_order=True, sample_rate=500, just_limb=False):
    
    samples= mat.shape[1]
    long=samples/sample_rate
    if time==None:
        time=(0,long)
    x=(np.arange(samples, )/sample_rate)[:samples]
    y=mat/1000
    y_of_time=y[:,int(time[0]*sample_rate):int(time[1]*sample_rate)]
    maxy=y_of_time.max()
    miny=y_of_time.min()
    
    if custom_order == True:
        y_new=np.empty((12,samples))
        y_new[0,:]=mat[4,:]
        y_new[1,:]=mat[0,:]
        y_new[2,:]=-mat[3,:]
        y_new[3,:]=mat[1,:]
        y_new[4,:]=mat[5,:]
        y_new[5,:]=mat[2,:]
        if just_limb==False:   
            y_new[6:12,:]=mat[6:12,:]
            
        leads =['aVL','I','-aVR','II','aVF','III','V1','V2','V3','V4','V5','V6']
        mat=y_new
    else:    
        leads=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']     

    mat=mat/1000    
    if just_limb==False:   
        fig, ax = plt.subplots(6,2) 
        for i in range (6):
            for j in range (2):
                y=mat[i+6*j,:]
                ax[i,j].plot(x,y)
                ax[i,j].set_ylabel (leads[i+6*j]  )              

                small_vertical=np.arange(0,long ,0.04)  
                big_vertical=np.arange(0,long,0.2)

                [ax[i,j].axvline(x=k, linestyle='--',linewidth=0.2 ) for k in small_vertical]
                [ax[i,j].axvline(x=k, linestyle='--',linewidth=0.5 ) for k in big_vertical]

                small_horizontal=np.arange(-3,+3 ,0.1) 
                big_horizontal=np.arange(-3,+3,0.5)
                [ax[i,j].axhline(y=k, linestyle='--',linewidth=0.2 ) for k in small_horizontal]
                [ax[i,j].axhline(y=k, linestyle='--',linewidth=0.5 ) for k in big_horizontal]
                ax[i,j].axis([time[0], time[1],miny,maxy])
    else:
        fig, ax = plt.subplots(6)
        j=0
        mat=mat[:6,:]
        maxy=y_of_time.max()
        miny=y_of_time.min()
        for i in range (6):
            y=mat[i,:]
            ax[i].plot(x,y)
            ax[i].set_ylabel(leads[i] , fontdict={'fontfamily':'serif' ,'fontsize':11, })
            # ax[i].set_yticklabels(ax[i].get_xticks(), rotation=0, size=9)
            # if i != 5:
            #     ax[i].set_xticks([])
            
            small_vertical=np.arange(0,long ,0.04)  
            big_vertical=np.arange(0,long,0.2)
            [ax[i].axvline(x=k, linestyle='--',linewidth=0.2 ) for k in small_vertical]
            [ax[i].axvline(x=k, linestyle='--',linewidth=0.5 ) for k in big_vertical]

            small_horizontal=np.arange(-3,+3 ,0.1) 
            big_horizontal=np.arange(-3,+3,0.5)
            [ax[i].axhline(y=k, linestyle='--',linewidth=0.2 ) for k in small_horizontal]
            [ax[i].axhline(y=k, linestyle='--',linewidth=0.5 ) for k in big_horizontal]
            ax[i].axis([time[0], time[1],miny,maxy])

    plt.subplots_adjust(left=0.11, bottom=0.1, right=0.95, top=0.95, wspace=.12, hspace=0.22)       



def subplot (channels , ecg , mat):
    leads=['I','II','III','aVR','aVL','aVF'
           ,'V1','V2','V3','V4','V5','V6']
    x=np.arange(ecg.samples, )/ecg.sample_rate
    
    if type(channels)!=list and channels!=None :
        plt.plot(x,mat [channels ,:]/1000)
        plt.ylabel(leads[channels])
    else:    
        if channels==None:
            channels=[i for i in range(12)]
    
        x=np.arange(ecg.samples, )/ecg.sample_rate
        num=len(channels)
        fig, ax = plt.subplots(num)

        for i in range (num):
            
            y=mat[channels[i],:]/1000
            maxy=y.max()
            miny=y.min()
            tool=ecg.duration
            small_vertical=np.arange(0,tool ,0.04)  
            big_vertical=np.arange(0,tool,0.2)

            [ax[i].axvline(x=j, linestyle='--',linewidth=0.1 ) for j in small_vertical]
            [ax[i].axvline(x=j, linestyle='--',linewidth=0.5 ) for j in big_vertical]

            small_horizontal=np.arange(miny,maxy ,0.1) 
            big_horizontal=np.arange(miny,maxy,0.5)
            [ax[i].axhline(y=j, linestyle='--',linewidth=0.1 ) for j in small_horizontal]
            [ax[i].axhline(y=j, linestyle='--',linewidth=0.5 ) for j in big_horizontal]
            ax[i].plot(x,y)
            ax[i].set_ylabel (leads[channels[i]])
            plt.subplots_adjust(left=0.11, bottom=0.02, right=0.98, top=0.98, wspace=0, hspace=0)
            
        fig.suptitle(f'index:{ecg.index} , age:{ecg.age} , sex:{ecg.sex}  ' , fontsize=9 )


def find_normals(folder_number=5):
    location=get_location(folder_number)[0]
    number_of_files=int (len([name for name in os.listdir(location) 
                              if os.path.isfile(os.path.join(location, name))])/2)
    count=0
    normal_numbers=[]
    for i in range (1,number_of_files+1):
        ecg=read_ecg(i , folder_number)
        if ecg.dx==['426783006']:
            normal_numbers.append(i)
            count+=1
    return normal_numbers , count    