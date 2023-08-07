import os
import numpy as np
import torch
from torchvision import transforms
import glob
from PIL import Image
import random
from tqdm import tqdm
from scipy import io


class UltrasoundDataset(torch.utils.data.Dataset):
    def __init__(self, mode, cfgs, phantom = 'all'):
        self.mode = mode
        self.datatype = cfgs['datatype']
        data_list = cfgs[mode + '_list']
        self.mat_files = []
        #Breastphantom
        if phantom == 'all' or phantom == 'breast':
            for data in data_list:
                scan_dir = os.path.join('/home/jaxa/Datasets/PlaneWaveImaging', str(cfgs['date']), 'Breastfan/IQdata', '{0:04}'.format(data + 1))
                self.mat_files.extend(sorted(glob.glob(os.path.join(scan_dir, '*'))))
        # #Evalphantom
        if phantom == 'all' or phantom == 'eval':
            eval_data_list = cfgs['eval_scan'][mode]
            for eval_data in eval_data_list:
                scan_dir = os.path.join('/home/jaxa/Datasets/PlaneWaveImaging', str(cfgs['date']), 'Evalfan/IQdata', '{0:04}'.format(int(eval_data)))
                self.mat_files.extend(sorted(glob.glob(os.path.join(scan_dir, '*'))))
        #invivo
        if phantom == 'all' or phantom == 'invivo':
            invivo_data_list = cfgs['invivo_scan'][mode]
            for invivo_data in invivo_data_list:
                scan_dir = os.path.join('/home/jaxa/Datasets/PlaneWaveImaging', str(cfgs['date']), 'invivo/IQdata', '{0:04}'.format(int(invivo_data)))
                self.mat_files.extend(sorted(glob.glob(os.path.join(scan_dir, '*'))))     

    def __getitem__(self, idx):
        mat_file = self.mat_files[idx]
        if self.datatype == 'rf':
            raw_rf_real, rf_real = self.load_data(mat_file, 'rf_real', 'input')
            raw_rf_imag, rf_imag = self.load_data(mat_file, 'rf_imag', 'input')
            input = np.stack([rf_real, rf_imag])
            raw_input_data = np.stack([raw_rf_real, raw_rf_imag])
            raw_rf_real, rf_real = self.load_data(mat_file, 'rf_real', 'output')
            raw_rf_imag, rf_imag = self.load_data(mat_file, 'rf_imag', 'output')
            output = np.stack([rf_real, rf_imag])
            raw_output_data = np.stack([raw_rf_real, raw_rf_imag])
        else:
            raw_input_data, input = self.load_data(mat_file, self.datatype, 'input')
            raw_output_data, output = self.load_data(mat_file, self.datatype, 'output')
        data = dict(gt=output, input=input, raw_gt=raw_output_data, raw_input=raw_input_data, input_path=mat_file)
        return data

    def load_data(self, mat_file, datatype, inout):
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
        data = transforms.Pad([2, 10], padding_mode='reflect')(data)
        return raw_data, data

    def __len__(self):
        return len(self.mat_files)

class PICMUSDataset(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        self.mat_files = []
        self.mat_files.extend(sorted(glob.glob(os.path.join('/home/jaxa/Datasets/PlaneWaveImaging/PICMUS', '*'))))

    def __getitem__(self, idx):
        mat_file = self.mat_files[idx]
        raw_rf_real, rf_real = self.load_data(mat_file, 'rf_real', 'input', 'real_input_rf')
        raw_rf_imag, rf_imag = self.load_data(mat_file, 'rf_imag', 'input', 'imag_input_rf')
        input = np.stack([rf_real, rf_imag])
        raw_input_data = np.stack([raw_rf_real, raw_rf_imag])
        raw_rf_real, rf_real = self.load_data(mat_file, 'rf_real', 'output', 'real_gt_rf')
        raw_rf_imag, rf_imag = self.load_data(mat_file, 'rf_imag', 'output', 'imag_gt_rf')
        output = np.stack([rf_real, rf_imag])
        raw_output_data = np.stack([raw_rf_real, raw_rf_imag])
        data = dict(gt=output, input=input, raw_gt=raw_output_data, raw_input=raw_input_data, input_path=mat_file)
        return data
    
    def load_data(self, mat_file, datatype, inout, dataform):
        if inout == 'input':
            raw_data = io.loadmat(os.path.join(mat_file, datatype, '0038.mat'))['{}'.format(dataform)].astype(np.float32)
        elif inout == 'output':
            raw_data = io.loadmat(os.path.join(mat_file, datatype, 'comp_{}.mat'.format(datatype)))['{}'.format(dataform)].astype(np.float32)
            
        if datatype == 'rf_real' or datatype == 'rf_imag':
            data = raw_data / np.max(np.abs(raw_data))
            data = transforms.ToTensor()(data)
        data = transforms.Pad([14, 15, 15, 16], padding_mode='reflect')(data)
        
        return raw_data, data
    
    def __len__(self):
        return len(self.mat_files)