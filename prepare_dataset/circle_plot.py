from cProfile import label
from math import e
import os
from turtle import circle
from scipy import io
import numpy as np
import torch as torch
import json
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as nd

set_num = 6
count = 0
r_sum = 0
save_data = dict()
save_coord = []
save_obj = []
index = []
for k in range(100):
    data_num =  k + 1
    data_path = '/home/jaxa/Datasets/PlaneWaveImaging/20230118/Evalfan/IQdata/' + '{:04}'.format(set_num) + '/' + '{:04}'.format(data_num)
    save_path = '/home/jaxa/shidara/PWI/src/prepare_dataset/results/' + '{:04}'.format(set_num)
    os.makedirs(os.path.join(save_path),exist_ok=True)
    eval_info = json.load(open('annotation_new.json', 'r'))
    coord_info = eval_info['/home/jaxa/Datasets/PlaneWaveImaging/20230118/Evalfan/IQdata/' + '{:04}'.format(set_num) + '/'+'{:04}'.format(data_num)]

    real = io.loadmat(os.path.join(data_path,'rf_real', 'comp_rf_real.mat'))['comp_rf_real'].astype(np.float32)
    imag = io.loadmat(os.path.join(data_path,'rf_imag', 'comp_rf_imag.mat'))['comp_rf_imag'].astype(np.float32)
    envelope = real + imag * 1j
    envelope = torch.from_numpy(envelope)
    envelope = torch.abs(envelope) / torch.max(abs(envelope))
    envelope = 20 * torch.log10(envelope).numpy()
    envelope[envelope < -60] = -60
    X, Y = np.meshgrid(np.arange(envelope.shape[1]), np.linspace(0, envelope.shape[0]-1, int(envelope.shape[0]/2)))
    
    envelope = cv2.remap(envelope, X.astype('float32'), Y.astype('float32'), cv2.INTER_CUBIC)
    fig1, ax1 = plt.subplots()
    ax1.imshow(envelope, 'gray')
    ax1.axis('off')

    for i, coord in enumerate(coord_info):
        for j, plot_info in enumerate(coord):
            if j == 1:
                one = np.ones(1,int)
                plot_info = np.concatenate([plot_info, one], 0)
                circle_info = plot_info
            elif j == 0:
                cx = plot_info[0]
                cy = plot_info[1]
            else:
                one = np.ones(1,int)
                plot_info = np.concatenate([plot_info, one], 0)
                circle_info = np.block([[circle_info], [plot_info]])
        theta = np.linspace(-np.pi, np.pi, 50)
        v = -(circle_info[:,0] ** 2 +  circle_info[:,1]** 2)
        u, residuals, rank, s = np.linalg.lstsq(circle_info, v, rcond=None)
        cx_pred = u[0] / (-2)
        cy_pred = u[1] / (-2)
        r_pred = np.sqrt(cx_pred ** 2 + cy_pred ** 2 - u[2])
        print(cy_pred)
        print(cx,cy,cx_pred, cy_pred, r_pred)
        count += 1
        r_sum += cy_pred
        plt.plot(cx_pred + 71 * np.cos(theta), cy_pred + 71 * np.sin(theta), 'r', label='least square')
        save_coord.append([cx_pred+2,cy_pred+5,50])
    plt.savefig('circle_plot'+'{:04}'.format(data_num)+'.png')
    plt.clf()
    plt.close()
    save_obj.append(save_coord)
    save_data[data_path] = save_obj
    with open(os.path.join('/home/jaxa/shidara/PWI/src/prepare_dataset/JSON/', 'NEWData_'+ '{:04}'.format(set_num) + '.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    save_obj, save_coord = [], []

print(r_sum/count)