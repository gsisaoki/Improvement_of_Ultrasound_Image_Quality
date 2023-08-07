import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from cProfile import label
import os
from scipy import io
import torch as torch
import json
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from lmfit import Parameters, Minimizer, report_fit

save_data = dict()
save_coord = []
save_obj = []

def fitting_2DGauss(data,W,H, j ):
    def two_D_gauss(X, A, sigma_x, sigma_y, mu_x, mu_y): # 2D Gaussian
        x, y = X
        z =  A * 1 / (np.sqrt(2*np.pi*sigma_x**2))*np.exp(-(x-mu_x)**2/(2*sigma_x**2)) * 1/(np.sqrt(2*np.pi*sigma_y**2))*np.exp(-(y-mu_y)**2/(2*sigma_y**2))
        return z
 
    def plot_fit_result(data, fit_result):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(fit_result["y"],fit_result["x"],fit_result["z"], rstride=10, cstride=10)  
        plt.show(block=False)
        plt.savefig('twoD_Gauss_fitting_'+'{:04}'.format(j+1)+'.png')
        plt.clf()
        plt.close()                                       
        return 0
        
    x_observed = data["x"]
    y_observed = data["y"]
    z_observed = data["z"]


    popt, pcov = curve_fit(two_D_gauss, (x_observed, y_observed), z_observed, bounds=([100,0,0,(W/2)-2,H/4], [5500,11,float('inf'), (W/2)+2, 3*H/4]),maxfev=50000000)
    o = z_observed #観測データ
    e = two_D_gauss((x_observed, y_observed), popt[0], popt[1], popt[2], popt[3], popt[4])
    
    residuals =  o - e 
    rss = np.sum(residuals**2)      
    tss = np.sum((o-np.mean(o))**2) 
    r_squared = 1 - (rss / tss) 
 
    print("*Result************")
    print("sigma_x = ", popt[1])
    print("sigma_y = ", popt[2])
    print("mu_x = ", popt[3])
    print("mu_y = ", popt[4])
    print("R^2 = ", r_squared)
    print("*******************")

    fit_x = np.linspace(min(data["x"]), max(data["x"]), 100)
    fit_y = np.linspace(min(data["y"]), max(data["y"]), 100)
    X, Y = np.meshgrid(fit_x, fit_y)
    fit_z = two_D_gauss((X, Y), popt[0], popt[1], popt[2], popt[3], popt[4])
    fit_result={"x":X, "y":Y, "z":fit_z}
    return popt[0], popt[1], popt[2], popt[3], popt[4]
    

set_num = 10

if (set_num == 5) or (set_num == 6) or (set_num == 7)or (set_num == 8)or (set_num == 11)or (set_num == 12)or (set_num == 13)or (set_num == 14):
    print("skip")
else:    
    for b in range(100):
        data_num = b + 1
        print(data_num)
        data_path = '/home/jaxa/Datasets/PlaneWaveImaging/20230118/Evalfan/IQdata/' + '{:04}'.format(set_num) + '/' + '{:04}'.format(data_num)
        save_path = '/home/jaxa/shidara/PWI/src/prepare_dataset/results/' + '{:04}'.format(set_num)
        os.makedirs(os.path.join(save_path),exist_ok=True)
        eval_info = json.load(open('annotation_new.json', 'r'))
        coord_info = eval_info['/home/jaxa/Datasets/PlaneWaveImaging/20230118/Evalfan/IQdata/' + '{:04}'.format(set_num) + '/'+'{:04}'.format(data_num)]
        
        # window params
        temp_scale_W = 5
        temp_scale_H = 5
        
        real = io.loadmat(os.path.join(data_path,'rf_real', 'comp_rf_real.mat'))['comp_rf_real'].astype(np.float32)
        imag = io.loadmat(os.path.join(data_path,'rf_imag', 'comp_rf_imag.mat'))['comp_rf_imag'].astype(np.float32)
        envelope = real + imag * 1j
        envelope = torch.from_numpy(envelope)
        envelope = torch.abs(envelope) / torch.max(abs(envelope))
        envelope60 = 20 * torch.log10(envelope)
        envelope = 20 * torch.log10(envelope)
        envelope60[envelope60 < -60] = -60 
        envelope[envelope < -30] = -30
        envelope60 = envelope60.numpy()
        envelope = envelope.numpy()
        X, Y = np.meshgrid(np.arange(envelope60.shape[1]), np.linspace(0, envelope60.shape[0]-1, int(envelope60.shape[0]/2)))
        envelope60 = cv2.remap(envelope60, X.astype('float32'), Y.astype('float32'), cv2.INTER_CUBIC)
        X, Y = np.meshgrid(np.arange(envelope.shape[1]), np.linspace(0, envelope.shape[0]-1, int(envelope.shape[0]/2)))
        envelope = cv2.remap(envelope, X.astype('float32'), Y.astype('float32'), cv2.INTER_CUBIC)
        fig1, ax1 = plt.subplots()
        ax1.imshow(envelope60, 'gray')
        ax1.axis('off')

        x ,y = np.meshgrid(np.linspace(0,2*temp_scale_W-1,2*temp_scale_W,dtype = int),np.linspace(0,2*temp_scale_H-1,2*temp_scale_H,dtype = int))

        for i, coord in enumerate(coord_info):
            for j, plot_info in enumerate(coord):
                replot_info = []
                temp_plot = []
                if plot_info[1] > envelope.shape[0]-temp_scale_H:
                    plot_info[1] = envelope.shape[0]-temp_scale_H
                elif plot_info[0] < temp_scale_W:
                    plot_info[0] = temp_scale_W
                elif plot_info[1] < temp_scale_H:
                    plot_info[1] = temp_scale_H
                elif plot_info[0] > envelope.shape[1]-temp_scale_W:
                    plot_info[0] = envelope.shape[1]-temp_scale_W
                temp_plot = [torch.div(envelope[plot_info[1]-temp_scale_H:plot_info[1]+temp_scale_H, plot_info[0]-temp_scale_W:plot_info[0]+temp_scale_W].argmax(), 2*temp_scale_W, rounding_mode='trunc'), envelope[plot_info[1]-temp_scale_H:plot_info[1]+temp_scale_H, plot_info[0]-temp_scale_W:plot_info[0]+temp_scale_W].argmax() % (2*temp_scale_W)]
                z = 30 + envelope[plot_info[1]-temp_scale_H:plot_info[1]+temp_scale_H, plot_info[0]-temp_scale_W:plot_info[0]+temp_scale_W]
                data={  "x":x.flatten(),
                        "y":y.flatten(),
                        "z":z.flatten()
                        }
                popt = fitting_2DGauss(data,2*temp_scale_W,2*temp_scale_H, j)
                
                replot_info = np.round([plot_info[1] + popt[4] - temp_scale_H, plot_info[0] + popt[3] - temp_scale_W])
                temp_img = envelope[plot_info[1]-temp_scale_H:plot_info[1]+temp_scale_H, plot_info[0]-temp_scale_W:plot_info[0]+temp_scale_W]
                
                save_coord.append([int(replot_info[1]),int(replot_info[0])])
                ax1.plot(replot_info[1], replot_info[0], 'r.', label= "after")
                
                ax1.text(replot_info[1], replot_info[0] - 5, str(j+1), color='r')
        plt.savefig(os.path.join(save_path, 'comp_' + '{:04}'.format(set_num) + '_' + '{:04}'.format(data_num)+'_replot_gaussian.png'))
        plt.clf()
        plt.close()
        save_obj.append(save_coord)
        save_data[data_path] = save_obj
        with open(os.path.join('/home/jaxa/shidara/PWI/src/prepare_dataset/JSON/', 'NEWData_'+ '{:04}'.format(set_num) + '.json'), 'w') as f:
            json.dump(save_data, f, indent=2)
        save_obj, save_coord = [], []

