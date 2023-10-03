from reading_dataset import *
from ecg2vcg import leads2vcg, limb2augmented
import statistics
import math
import scipy
from scipy.signal import butter, filtfilt
from denoise_wavelet import denoising_data
from tqdm import tqdm



def unify_ecg (mat): 
        
        mat_I = mat[0,:]
        mat_II = mat[1,:]
        mat_aVR = mat[3,:]
        mat_aVL = mat[4,:]
        mat_aVF = mat[5,:]

        _,_,theta,r= leads2vcg(mat_I,mat_II , 0,-np.pi/3)
        mat[2,:]=limb2augmented(theta,r,-2*np.pi/3)
        mat[3,:]=limb2augmented(theta,r,5*np.pi/6)
        mat[4,:]=limb2augmented(theta,r,np.pi/6)
        mat[5,:]=limb2augmented(theta,r,-np.pi/2)

        return mat

def ecg2twoD (mat,time0=None,time1=None ,just_limb=True):
    
    if time0 == None and time1==None:
        time0=0
        time1=len(mat[0])
    mat_I = mat[0,time0:time1]
    mat_II = mat[1,time0:time1]
    mat_III = mat[2,time0:time1]
    mat_aVR = mat[3,time0:time1]
    mat_avl = mat[4,time0:time1]
    mat_aVF = mat[5,time0:time1]
    if just_limb==False:
        V1=mat[6,time0:time1]
        V2=mat[7,time0:time1]
        V3=mat[8,time0:time1]
        V4=mat[9,time0:time1]
        V5=mat[10,time0:time1]
        V6=mat[11,time0:time1]
    A=[mat_I,mat_II,mat_III,mat_aVR,mat_avl,mat_aVF]
    B=['I','II','III','aVR','aVL','aVF']
    thetas=[0,-np.pi/3,-np.pi*2/3 ,np.pi*5/6,np.pi/6,-np.pi/2]
    Coordinates=[]
    for i in range(len(A)):
        for j in range(len(A)):
            if i!=j:
                Coordinates.append((leads2vcg(A[i],A[j],thetas[i],thetas[j]), f'{B[i]},{B[j]}'  ))
    return Coordinates
# ecg2twoD gives you a list which contains 30 tuple 
# in each tuple you have another tuple (that we call it tupleB) and a string
# this tupleB contains 4 matrices which are X , Y , THETA and R
# the string shows the pair leads that this coordinates are given from that pair :) 
# confused ? me to. sorry :)
# coordinate[j][0][0][i] = X[i] which i represents the length and j is between 1 to 30 and represents the pair leads (I,aVL for example)

def convert_xs_2_Weighted_mean (list_of_xs):
    # average=sum(list_of_xs)/6
    var_list_of_xs=statistics.variance(list_of_xs)
    zarayeb=[]
    if var_list_of_xs !=0:
        for i in range (len (list_of_xs)) :
            b=list_of_xs[0:i]+list_of_xs[i+1:]
            var_without_item = statistics.variance(b)
            zarayeb.append(var_without_item/var_list_of_xs)
        new_list=[]    
        # zarayeb=list(map(lambda x:x**2 , zarayeb))
        for i in range (len(list_of_xs))  :
            new_list.append(list_of_xs[i]*zarayeb[i])

        new_average=sum(new_list)/sum(zarayeb)
    else:
        new_average= sum(list_of_xs)/len(list_of_xs)
    return new_average   

# def mean_denoise (coordinates):
#     lenght=len(coordinates[0][0][0])

#     x_kol=[]
#     y_kol=[]
#     for i in tqdm(range(lenght)) :
#         X=[]
#         Y=[]
#         for j in range (30):
#             x=coordinates[j][0][0][i]
#             y=coordinates[j][0][1][i]
#             X .append(x)
#             Y .append(y)
#         x_new=convert_xs_2_Weighted_mean(X)   
#         y_new=convert_xs_2_Weighted_mean(Y) 
#         x_kol.append(x_new)
#         y_kol.append(y_new)
#     x_kol=np.array(x_kol)
#     y_kol=np.array(y_kol)

#     return (x_kol,y_kol) 
            



def mean_denoise (coordinates):
    x_kol=[]
    y_kol=[]
    

    lenght=len(coordinates[0][0][0])
    for i in range (lenght):
        X=[]
        Y=[]
        small_dict_x={0:[],1:[],2:[],3:[],4:[],5:[]}
        small_dict_y={0:[],1:[],2:[],3:[],4:[],5:[]}
        # used_leads=[]
        list_of_6leadsX=[0,0,0,0,0,0]
        list_of_6leadsY=[0,0,0,0,0,0]
        for j in range (30):
            # two_label=list_of_3taee[j][1]
            # first_label=re.search('(.*),',two_label)
            # second_label=re.search(',(.*)',two_label)
            # if (second_label,first_label) in used_leads :
            #     continue
            # used_leads.append((first_label,second_label))
            x=coordinates[j][0][0][i]
            y=coordinates[j][0][1][i]
            X .append(x)
            Y .append(y)
            a= int(j/5)
            small_dict_x[a].append(x)
            small_dict_y[a].append(y)
            list_of_6leadsX[a]+=x
            list_of_6leadsY[a]+=y

        for item in small_dict_x:
            small_dict_x[item]=convert_xs_2_Weighted_mean(small_dict_x[item])
            small_dict_y[item]=convert_xs_2_Weighted_mean(small_dict_y[item])
        list_of_6leadsX = list(small_dict_x.values())
        list_of_6leadsY = list(small_dict_y.values())
    

        # list_of_6leadsX=list(map(lambda x:x/5,list_of_6leadsX))
        # list_of_6leadsY=list(map(lambda x:x/5,list_of_6leadsY))



        new_x=convert_xs_2_Weighted_mean(list_of_6leadsX)
        new_y=convert_xs_2_Weighted_mean(list_of_6leadsY)
        # mean_of_all_leads_X= sum(X)/len(X)
        # mean_of_all_leads_Y= sum(Y)/len(Y)
        x_kol.append(new_x)
        y_kol.append(new_y)
    x_kol=np.array(x_kol)
    y_kol=np.array(y_kol)

    return (x_kol,y_kol)   


def reproduce_leads_from_denoise(x,y, mat):
    a=np.zeros(mat.shape)
    a[6:11,:]=mat[6:11,:]
    thetas=[0,-np.pi/3,-np.pi*4/6 ,np.pi*5/6,np.pi/6,-np.pi/2]
    for i in range (len(thetas)):
        theta=thetas[i]
        leads=[]
        for j in range(len(x)):
            if x[j]<0 and y[j]>0 or x[j]<0 and y[j]<0:
                alpha=np.pi+math.atan(y[j]/x[j])
            else :
                alpha=math.atan(y[j]/x[j])
            r=math.sqrt(x[j]**2+y[j]**2)
            betha=theta-alpha
            lead_volt=r*math.cos(betha)
            leads.append(lead_volt)
        lead=np.array(leads)    
        a[i,:]=lead   

    return a   




class Main_elements ():
    def __init__(self, clean_method, noise_method, denoise_method):
        self.clean_method=clean_method
        self.noise_method=noise_method
        self.denoise_method=denoise_method



