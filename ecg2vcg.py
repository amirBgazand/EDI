from cProfile import label
from turtle import color
from numpy.lib.type_check import real
from reading_dataset import *
import regex as re
import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import math
import pickle 
from mpl_toolkits import mplot3d
import warnings
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d

# warnings.filterwarnings('error')
plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.size"] = 11


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)



def rotate(x,y,theta):
    rotation_matrix=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    newarray=np.empty((2,x.shape[0]))
    newarray[0,:]=x
    newarray[1,:]=y
    newarray=rotation_matrix.dot(newarray)
    x_new=newarray[0,:]
    y_new=newarray[1,:]
    return (x_new , y_new)



def leads2vcg(first_mat,second_mat,theta1,theta2):

    
    second_mat=second_mat.astype('float64')
    first_mat=first_mat.astype('float64')
    second_mat[second_mat==0]=0.01
    nesbat=first_mat/second_mat
    
    soorat=nesbat*np.cos(theta2)-np.cos(theta1)
    makhraj=np.sin(theta1)-nesbat*np.sin(theta2)
    makhraj[makhraj==0]=0.01

    theta=np.arctan(soorat/makhraj)
    
    r=first_mat/(np.cos(theta-theta1))
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    for i in range (len(x)):
        if x[i]==0:
            x[i]=x[i-1]

    for i in range (len(y)):
        if y[i]==0:
            y[i]=y[i-1] 
    
    return x, y,theta,r



def limb2augmented(theta,r,theta1):
 
    augmented=r*(np.cos(theta-theta1))
       
    return augmented  


def get_peak () :
    peaks=pickle.load (open(f'{pkl_folder}peaks.pkl' , 'rb'))



def threeD (x,y,z):
    max_x=np.max(x)
    max_y=np.max(y)
    min_x=np.min(x)
    min_y= np.min(y)
    max=np.max([max_x,max_y])
    min=np.min([min_x,min_y])
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'gray')
    ax.set_xlim([min,max])
    ax.set_ylim([min,max])




def plot(label,ro , t,ax, enteha):
    scale=10
    color1='grey'
    color2='black'
    width=1.5
    tetha=np.pi/6 if label=='aVL' else  0 if label=='I' else -np.pi/6 if label=='-aVR'   else  -np.pi/3 if label=='II' else -np.pi/2 if label=='aVF' else -np.pi*4/3 
    if label == 'aVF' :
        
        x=np.linspace(0,ro,100)
        p=plt.plot(x*0,x*0+enteha,-x ,'--',linewidth=width, label='aVF' ,color=color1)
        # ax.arrow3D(0,enteha,0,0,0,-2,
        #                 mutation_scale=scale,
        #                 arrowstyle="-|>",color=color1,
        #                 linewidth=1)
        # ax.arrow3D(0,enteha,0,1,0,1,
        #                 mutation_scale=scale,
        #                 arrowstyle="-|>",color=color1,
        #                 linewidth=1)
        # color=p[0].get_color()
        # ax.text((x*0)[-10],((x*0)+enteha)[-1],-x[-10], s=f'{label}',color=color2)
    else:
        maximum=np.sqrt(((ro**2)/(1+(np.tan(tetha))**2)))
        x=np.linspace(0,maximum,100)
        y=x*np.tan(tetha)
        if label=='III':
            p=plt.plot (-x,x*0+enteha , y , '--' ,linewidth=width , label=label,color=color1)
            # color=p[0].get_color()
            # ax.text(-x[-10],((x*0)+enteha)[-1], y[-10], s=f'{label}',color=color2)

        else:    
            p=plt.plot (x,x*0+enteha , y , '--' ,linewidth=width , label=label,color=color1 )
            # color=p[0].get_color() 
            # ax.text(x[-10],((x*0)+enteha)[-1], y[-10], s=f'{label}' , color=color2)


def plot_xy(ro ,ax):
    x=np.linspace(0,ro,100)
    plt.plot(x*0,x*0,x ,linewidth=0.5, label='Y' , color= 'black' , )
    # ax.text((x*0)[-10],(x*0)[-1],x[-10], s=f'Y' , color='black')

    plt.plot (x,x*0 , x*0  ,linewidth=0.5 , label='X', color='black')
    # ax.text(x[-1],(x*0)[-1], (x*0)[-1], s=f'X', color='black')





def ecg2vcgplot(index,mat=[],peak_number=0,howmany_peaks=3,folder_number=5 , type='all' , denoise=False,sixSecond=False):
        ecg= read_ecg(index , folder_number )
        peaks=get_peak()
        ecg_peaks=peaks[index]
        my_peak=0.1
        next_peak=ecg_peaks[peak_number+howmany_peaks]-0.6
        if sixSecond==True:
            my_peak=0
            next_peak=6
        if len(ecg_peaks)-1 < howmany_peaks :
            print(ecg.index)
        else:  
            
            time0=int(my_peak*ecg.sample_rate)
            time1=int(next_peak*ecg.sample_rate)
            
            duration=np.arange(ecg.samples, )/ecg.sample_rate
            for_mehvar=duration[0:time1-time0+100]
            duration=duration [0:time1-time0]
            if mat==[]:
                mat=ecg.mat
                denoising_mode=False
            else:
                denoising_mode=True    
                

            mat_I = mat[0,time0:time1]/1000
            mat_II = mat[1,time0:time1]/1000
            mat_III = mat[2,time0:time1]/1000
            mat_aVR = mat[3,time0:time1]/1000
            mat_avl = mat[4,time0:time1]/1000
            mat_aVF = mat[5,time0:time1]/1000

            # V1=ecg.mat[6,time0:time1]
            # V2=ecg.mat[7,time0:time1]
            # V3=ecg.mat[8,time0:time1]
            # V4=ecg.mat[9,time0:time1]
            # V5=ecg.mat[10,time0:time1]
            # V6=ecg.mat[11,time0:time1]



            x1,y1=rotate(mat_II,mat_avl,-np.pi/3 )  
            x2,y2=rotate(mat_aVF,mat_I,-np.pi/2)
            x3,y3=rotate(mat_aVR,mat_III,5*np.pi/6)

            x4,y4,_,_= leads2vcg(mat_I,mat_II , 0,-np.pi/3)
            x5,y5,_,_=leads2vcg(mat_I,mat_III , 0,-4*np.pi/6)
            x6,y6,_,_=leads2vcg(mat_II,mat_III , -np.pi/3,-4*np.pi/6)
            averages=[]
            for i in range (len(x4)):
                x4_and_x5=np.sqrt((x4[i]-x5[i])**2+(y4[i]-y5[i])**2) 
                x4_and_x6=np.sqrt((x4[i]-x6[i])**2+(y4[i]-y6[i])**2)
                x5_and_x6=np.sqrt((x5[i]-x6[i])**2+(y5[i]-y6[i])**2)
                average=(x4_and_x5+x4_and_x6+x5_and_x6)/3
                averages.append(average)
            print(f'average for {index} = {sum(averages)/len(averages)}')

            # x7,y7=leads2vcg(mat_avl,-mat_aVR , np.pi/6,-np.pi/6)
            # x8,y8=leads2vcg(-mat_aVR,mat_aVF , -np.pi/6,-np.pi/2)
            # x9,y9,_,_=leads2vcg(mat_avl,mat_aVF , np.pi/6,-np.pi/2)

            # return [(x4,y4),(x5,y5),(x6,y6)]


            # x10,y10=leads2vcg(V4,V5 , -np.pi/3,-np.pi/6)
            # x11,y11=leads2vcg(V5,V6 , -np.pi/6,0)
            # x12,y12=leads2vcg(V1,V2 , -np.pi*100/180,-np.pi*80/180)
            # x13,y13=leads2vcg(V3,V2 , -np.pi*75/180,-np.pi*80/180)
            # x14,y14=leads2vcg(V4,V2 , -np.pi/3,-np.pi*80/180)
            # x15,y15=leads2vcg(V1,V6 , -np.pi*100/180,0)
            # x16,y16=leads2vcg(V1,V5 , -np.pi*100/180,-np.pi/6)
            
            # x17,y17=leads2vcg(V4,V6 , -np.pi/3,0)

            
            # plt.figure()
            # if type== 'safhe':
                
                # plt.plot(x6,y6 )
                # plt.plot(x4,y4,label= 'I,II')
                # plt.plot(x5,y5,label= 'I,III')
                # plt.plot(x8,y8 , label= '-aVR,aVF')
                # plt.plot(x9,y9 , label= 'aVL,aVF')
                # plt.plot(x5,y5 , label='I,III' ,color='blu
                # plt.show()
                
        
                      
                # 
                #                                                           
                                                        

            # if type == 'old 3d':
            #     max_x=np.max(x1)
            #     max_y=np.max(y1)
            #     min_x=np.min(x1)
            #     min_y= np.min(y1)
            #     max=np.max([max_x,max_y])
            #     min=np.min([min_x,min_y])
            #     ax = plt.axes(projection='3d')
            #     plt.plot(x4,y4,duration ,  label= 'I,II')
            #     plt.plot(x5 ,y5,duration,  label='I,III')
            #     plt.plot(x6,y6,duration ,  label= 'II,III')
            #     plt.plot(x7 ,y7, duration, label= 'aVL,-aVR')
            #     plt.plot(x8, y8,duration , label= '-aVR,aVF')
            #     plt.plot(x9,y9,duration, label='avl.avf')
            #     ax.set_xlim([min-(max-min)/2,max+(max-min)/2])
            #     ax.set_ylim([min-(max-min)/2,max+(max-min)/2])

            if type == '3d':

                max_x=np.max(x1)
                max_y=np.max(y1)
                min_x=np.min(x1)
                min_y= np.min(y1)
                maximum=np.max([max_x,max_y])
                minimum=np.min([min_x,min_y])
                ax = plt.axes(projection='3d')
                
                
                plt.plot(x4,duration ,y4,  label= 'curve by lead I,II' ,linewidth=2.5)
                # plt.plot(x5,duration ,y5,  label= 'curve by lead I,III')
                # plt.plot(x6,duration ,y6,  label= 'curve by lead II,III')
                
                # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                if denoising_mode ==False:
                    ranging=0.05
                    j=0
                    x_scatter=[]
                    y_scatter=[]
                    time_scatter=[]
                    for i in range (len(x4)):
                        
                        r_new=x4[i]**2+y4[i]**2
                        if i ==0:
                            x_scatter.append(x4[i])
                            y_scatter.append(y4[i])
                            r_old=r_new
                            i_old=i
                            time_scatter.append((i)/500)
                        else:
                            if  r_new > r_old+ranging or r_new < r_old-ranging or i-i_old>30 :
                                x_scatter.append(x4[i])
                                y_scatter.append(y4[i])
                                r_old=r_new
                                i_old=i
                                time_scatter.append((i)/500)
                                

                        # if x4[i+1]^2+y4[i+1]^2 >ranging+x4[j]^2+y4[j]^2:
                    
                    ax.scatter(x_scatter,time_scatter ,y_scatter,  label= 'I,II' , facecolors='none', edgecolors='r',linewidth=1.5, s=25)
                
                ro=np.max([abs(maximum),abs(minimum)])*1.2
                max_x=np.arange(int(2*ro*1000))/1000
                max_y=np.arange(int(2*ro*1000) )/1000 
                max_y=np.arange(int(2*2*1000) )/1000 

                # plt.axline((0, 0,0), (1, 1,1), linewidth=4, color='r')
                
                # fig = plt.figure()

                scale=15
                ax.arrow3D(0,0,0,ro*1.2,0,0,
                        mutation_scale=scale,
                        arrowstyle="-|>",color='black',
                        linewidth=1)
                ax.arrow3D(0,0,0,0,0,ro*1.2,
                        mutation_scale=scale,
                        arrowstyle="-|>",color='black',
                        linewidth=1)
                ax.arrow3D(0,0,0,0,2.1,0,
                        mutation_scale=scale,
                        arrowstyle="-|>",color='black',
                        linewidth=1)
                
                if denoising_mode==False:
                    ro =2
                    theta=np.linspace(0,2 * np.pi , 100)
                    x=ro*np.cos(theta)
                    y=ro*np.sin(theta)
                    enteha=1.75
                    plt.plot(x,x*0+enteha,y , color='gray' , linewidth=1.5 )

                    labels=['aVL','I','-aVR','II','aVF','III']
                    # plot_xy(2500,ax)
                    for label in labels:
                        plot(label, ro,next_peak,ax , enteha)

                
                adad=2.5
                ax.set_xlim(-adad,adad)
                ax.set_zlim(-adad,adad)

                # plt.plot((0,ro),(0,0),(0,0),color='black' , linewidth=1.2 )
                # plt.plot((0,0),(0,2),(0,0),color='black' , linewidth=1.2 )
                # plt.plot((0,0),(0,0),(0,ro),color='black' , linewidth=1.2 )

                # plt.plot(for_mehvar*0,for_mehvar ,for_mehvar*0 ,color='black' , linewidth=1.2)
                # plt.plot(max_x,max_x*0 ,max_x*0 ,color='black' , linewidth=1.2)
                # plt.plot(max_y*0,max_y*0 ,max_y ,color='black', linewidth=1.2)
                # plt.plot(duration,duration*0 ,duration ,color='black')
                # plt.plot(x5,duration ,y5,  label='I,III')
                # plt.plot(x6,duration ,y6,  label= 'II,III')
                # plt.plot(x7, duration ,y7, label= 'aVL,-aVR')
                # plt.plot(x8,duration , y8, label= '-aVR,aVF')
                # plt.plot(x9,duration,y9, label='avl.avf')
                
                '''
                # a=2

                # # theta = np.linspace(2 * np.pi,0 , 100)
                # # y = 10*np.cos(theta)
                # # z = 10*np.sin(theta)
                # # i=0
                # # phi = 10*np.pi/9
                # # plt.plot(y*np.sin(phi)+10*np.sin(phi),
                # #         y*np.cos(phi)+10*np.cos(phi),z,)
                '''

                ######Creating_circle
                

            #     ax.set_xlim([min-(max-min)/2,max+(max-min)/2])
            #     ax.set_zlim([min-(max-min)/2,max+(max-min)/2])
            #     # for ii in range(0,360,1):
            #     #     ax.view_init(elev=-ii, azim=0)
            #     #     plt.savefig("movie%d.png" % ii)
            #     ax.view_init(elev=15, azim=-15)
            #     # adad=rotation_fig(-60,30,-90,0,0,ax)
            #     # adad=rotation_fig(-90,0,-60,30,adad,ax)
            #     # adad=rotation_fig(-60,30,0,30,adad,ax)
            #     # adad=rotation_fig(0,30,0,-211,adad,ax)
                


            # if type == 'Vs'  :
            #     ax = plt.axes(projection='3d')
                # plt.plot(x10,y10,duration, label='V4,V5')
                # plt.plot(x11,y11,duration , label ='V5,V6')
                # plt.plot(x17,y17,duration , label ='V4,V6')
                # plt.plot(x12,y12,duration , label ='V1,V2')
                # plt.plot(x13,y13,duration , label ='V2,V3')
                # plt.plot(x14,y14,duration , label ='V2,V4')
                # plt.plot(x15,y15,duration , label ='V1,V6')  
                # plt.plot(x16,y16,duration , label ='V1,V5')


        # plt.legend(loc="upper left" , fontsize='12')
        # plt.suptitle(f'index:{ecg.index} , age:{ecg.age} , sex:{ecg.sex}\n , dx:{str(new)} ' , fontsize=9 )
        plt.subplots_adjust(left=0.00, bottom=0.00, right=1, top=1, wspace=0, hspace=0)
    






def ecg2vcg(index,peak_number=0,howmany_peaks=5,folder_number=5 , type='all' , denoise=False,sixSecond=False):
        ecg= read_ecg(index , folder_number , denoise=denoise)
        ecg_peaks=peaks[index]
        my_peak=ecg_peaks[peak_number]
        next_peak=ecg_peaks[peak_number+howmany_peaks]
        if sixSecond==True:
            my_peak=0
            next_peak=6
        if len(ecg_peaks)-1 < howmany_peaks :
            print(ecg.index)
        else:  
            
            time0=int(my_peak*ecg.sample_rate)
            time1=int(next_peak*ecg.sample_rate)
            
            duration=np.arange(ecg.samples, )/ecg.sample_rate
            duration=duration [time0:time1]

            mat_I = ecg.mat[0,time0:time1]
            mat_II = ecg.mat[1,time0:time1]
            mat_III = ecg.mat[2,time0:time1]
            mat_aVR = ecg.mat[3,time0:time1]
            mat_avl = ecg.mat[4,time0:time1]
            mat_aVF = ecg.mat[5,time0:time1]

            # V1=ecg.mat[6,time0:time1]
            # V2=ecg.mat[7,time0:time1]
            # V3=ecg.mat[8,time0:time1]
            # V4=ecg.mat[9,time0:time1]
            # V5=ecg.mat[10,time0:time1]
            # V6=ecg.mat[11,time0:time1]



            # x1,y1=rotate(mat_II,mat_avl,-np.pi/3 )  
            # x2,y2=rotate(mat_aVF,mat_I,-np.pi/2)
            # x3,y3=rotate(mat_aVR,mat_III,5*np.pi/6)

            x4,y4,_,_= leads2vcg(mat_I,mat_II , 0,-np.pi/3)
            x5,y5,_,_=leads2vcg(mat_I,mat_III , 0,-4*np.pi/6)
            x6,y6,_,_=leads2vcg(mat_II,mat_III , -np.pi/3,-4*np.pi/6)

            # x7,y7=leads2vcg(mat_avl,-mat_aVR , np.pi/6,-np.pi/6)
            # x8,y8=leads2vcg(-mat_aVR,mat_aVF , -np.pi/6,-np.pi/2)
            # x9,y9,_,_=leads2vcg(mat_avl,mat_aVF , np.pi/6,-np.pi/2)

            return [(x4,y4),(x5,y5),(x6,y6)]

    
# ecg2vcg(1,howmany_peaks=1,type='safhe')
# ecg2vcg(1,howmany_peaks=1,type='safhe')