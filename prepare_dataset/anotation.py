import cv2
import numpy as np
import os
import glob
from scipy import io
from torchvision import transforms
import torch
import json

data = dict()
coord = []
obj = []

temp = glob.glob(os.path.join('/local_disk/Datasets/PlaneWaveImaging/20230118/Eval/IQdata/*.mat'))
mat_files = []
for i in range(len(temp)):
    scan_dir = os.path.join('/home/jaxa/Datasets/PlaneWaveImaging/20230118/Evalfan/IQdata/', '{0:04}'.format(i + 1))
    mat_files.extend(sorted(glob.glob(os.path.join(scan_dir, '*'))))
    
for mat_path in mat_files:
    data[mat_path] = []

def click_pos(event, x, y, flags, params):
    global data, coord, obj
    if event == cv2.EVENT_LBUTTONDOWN:
        coord.append([x, y])
        img2 = np.copy(image)
        for temp in coord:
            cv2.circle(img2,center=(temp[0],temp[1]),radius=3,color=0,thickness=-1)
        cv2.imshow('window', img2)

    if event == cv2.EVENT_RBUTTONDOWN:
        obj.append(coord)
        coord = []
        cv2.imshow('window', image)
        
    if event == cv2.EVENT_MBUTTONDOWN:
        coord.clear()
        cv2.imshow('window', image)


def load_data(mat_file, datatype, inout):
    if inout == 'input':
        raw_data = io.loadmat(os.path.join(mat_file, datatype, '0038.mat'))['{}'.format(datatype)].astype(np.float32)
    elif inout == 'output':
        raw_data = io.loadmat(os.path.join(mat_file, datatype, 'comp_{}.mat'.format(datatype)))['comp_{}'.format(datatype)].astype(np.float32)
    if datatype == 'us':
        data = raw_data + np.abs(raw_data.min())
        data = transforms.ToTensor()(data) / 60
        data = transforms.Normalize((0.5), (0.5))(data)
    elif datatype == 'envelope':
        data = transforms.ToTensor()(raw_data)
    elif datatype == 'rf_real' or datatype == 'rf_imag':
        data = raw_data / np.max(np.abs(raw_data))
        data = transforms.ToTensor()(data)
    return raw_data, data

def transform2image(input, raw_data):
    real, imag = input[0, :, :] * torch.max(torch.abs(raw_data[0, :, :])), input[1, :, :] * torch.max(torch.abs(raw_data[1, :, :]))
    rf = torch.complex(real, imag)
    envelope = torch.abs(rf) / torch.max(torch.abs(rf))
    input = 20 * torch.log10(envelope)
    input[input < -40] = -40
    input = input + torch.abs(input.min())
    input = input / 40
    input = torch.unsqueeze(input, 1)
    return input

j = 0

while j != len(mat_files):
    mat_path = mat_files[j]
    print(mat_path)
    raw_rf_real, rf_real = load_data(mat_path, 'rf_real', 'output')
    raw_rf_imag, rf_imag = load_data(mat_path, 'rf_imag', 'output')
    output = torch.stack([rf_real, rf_imag])
    raw_output_data = np.stack([raw_rf_real, raw_rf_imag])
    
    image = torch.squeeze(transform2image(output, torch.tensor(raw_output_data))).numpy()
    
    X, Y = np.meshgrid(np.arange(image.shape[1]), np.linspace(0, image.shape[0]-1, int(image.shape[0]/2)))
    image = cv2.remap(image, X.astype('float32'), Y.astype('float32'), cv2.INTER_CUBIC)

    cv2.namedWindow("window", 16)
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.imshow('window', image)
    cv2.resizeWindow("window", 1000, int(1000 * image.shape[0] / image.shape[1])) 
    cv2.moveWindow('window', 500, 100)
    cv2.setMouseCallback('window', click_pos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    obj.append(coord)
    data[mat_path] = obj
    with open(os.path.join('/home/jaxa/shidara/PWI/src/prepare_dataset/JSON/', 'annotation_new_6.json'), 'w') as f:
        json.dump(data, f, indent=2)
    obj, coord = [], []

    print('Next Image?')
    val = input()
    if val == 'q':
        j += 1
    elif val == 's':
        j += 100
    else:
        print('Please retry anotation.')
