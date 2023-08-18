import numpy as np
import pandas as pd
from torchvision import transforms
import torch
import glob
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import h5py

import models.conventional.Li as Li
import models.proposed.Unet1D2D.model as proposedUnet
from criterion import *

def select_model(cfgs):
    if cfgs['model'] == 'unet2D':
        model = proposedUnet.Unet2D(in_channels=cfgs['channel'], classes=cfgs['channel'])
    if cfgs['model'] == 'unet1D':
        model = proposedUnet.Unet1D(in_channels=cfgs['channel'], classes=cfgs['channel'])
    if cfgs['model'] == 'unet1D2D':
        model = proposedUnet.Unet1D2D(in_channels=cfgs['channel'], classes=cfgs['channel'])
    if cfgs['model'] == 'unet1D2D_featuremap':
        model = proposedUnet.Unet1D2D_featuremap(in_channels=cfgs['channel'], classes=cfgs['channel'])    
    if cfgs['model'] == 'Li':
        model = Li.LiNet(in_channels=cfgs['channel'], classes=cfgs['channel'])
    return model

def return_zero_fill(num):
    return '{0:03d}'.format(num)

def save_image(input, gt, pred, save_path):
    fig = plt.figure(tight_layout=True)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(gt[:, :], 'gray');ax1.axis('off');ax1.set_title('Compound Image')
    ax2.imshow(input[:, :], 'gray');ax2.axis('off');ax2.set_title('Input Image')
    ax3.imshow(pred[:, :], 'gray');ax3.axis('off');ax3.set_title('Estimated Image')
    plt.savefig(save_path)
    plt.clf()
    plt.close()

def transform2image(input, raw_data, datatype):
    if datatype == 'us':
        input = input * 0.5 + 0.5
    elif datatype == 'envelope':
        input = 20 * torch.log10(input)
        input[input < -60] = -60
        input = input + torch.abs(input.min())
        input = input / 60
    elif datatype in ['rf_real', 'rf_imag']:
        input = input * torch.max(torch.abs(raw_data))
        input = torch.abs(input) / torch.max(torch.abs(input))
        input = 20 * torch.log10(input + 1e-10)
        input[input < -60] = -60
        input = input + torch.abs(input.min())
        input = input / 60
        input = torch.unsqueeze(input, 1)
    elif datatype == 'rf':
        real, imag = input[:, 0, :, :] * torch.max(torch.abs(raw_data[:, 0, :, :])), input[:, 1, :, :] * torch.max(torch.abs(raw_data[:, 1, :, :]))
        rf = torch.complex(real, imag)
        envelope = torch.abs(rf) / torch.max(torch.abs(rf))
        input = 20 * torch.log10(envelope)
        input[input < -60] = -60
        input = input + torch.abs(input.min())
        input = input / 60
        input = torch.unsqueeze(input, 1)
    return input

def transform2envelope(input, raw_data):
        real, imag = input[:, 0, :, :] * torch.max(torch.abs(raw_data[:, 0, :, :])), input[:, 1, :, :] * torch.max(torch.abs(raw_data[:, 1, :, :]))
        rf = torch.complex(real, imag)
        envelope = torch.abs(rf) / torch.max(torch.abs(rf))
        return envelope

def pixel_remap(input):
    X, Y = np.meshgrid(np.arange(input.shape[-1]), np.linspace(0, input.shape[-2]-1, int(input.shape[-2]/2)))
    input = cv2.remap(input.numpy(), X.astype('float32'), Y.astype('float32'), cv2.INTER_CUBIC)
    return input
    
def stack_image(input):
    input_list = []
    input_list.append(input)
    input_list.append(input)
    input_list.append(input)
    input = torch.stack(input_list, dim = 1)
    input = torch.squeeze(input)
    return input

def raylfit(x,data):
    sum = 0
    for i in range(len(data)):
        sum += data[i]**2
    b = np.sqrt(sum/(2*(len(data)+1)))
    return (x/b**2)*np.exp(-x**2/(2*b**2))

# Compute sn (CUBDL)
def signal_to_noise(data):
    return data.mean() / data.var()

# Compute contrast ratio (CUBDL)
def contrast_envelope(roi_data, bg_roi_data):
    return 20 * np.log10(roi_data.mean() / bg_roi_data.mean())

# Compute contrast ratio (Performance testing of medical ultrasound equipment:fundamental vs. harmonic mode)
def contrast_gray(roi_data, bg_roi_data):
    return 20 * np.log10(np.abs(roi_data.mean() - bg_roi_data.mean()) / ((roi_data.mean() + bg_roi_data.mean()) / 2))

# Compute contrast-to-noise ratio (CUBDL)
def contrast_to_noise_envelope(roi_data, bg_roi_data):
    return np.abs(roi_data.mean() - bg_roi_data.mean()) / np.sqrt((roi_data.var()**2 + bg_roi_data.var()**2))

# Compute contrast-to-noise ratio (PICMUS)
def contrast_to_noise_gray(roi_data, bg_roi_data):
    return 20 * np.log10(np.abs(roi_data.mean() - bg_roi_data.mean()) / np.sqrt(((roi_data.var()**2 + bg_roi_data.var()**2) / 2)))
    

# Compute gcnr (CUBDL)
def generalized_contrast_to_noise(roi, bg_roi):
    _, bins = np.histogram(np.concatenate((roi, bg_roi)), bins=256)
    f, _ = np.histogram(roi, bins=bins, density=True)
    g, _ = np.histogram(bg_roi, bins=bins, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))

def cr_prop_fn(roi, roi_gray, bg_roi, bg_roi_gray):
    cr = ((np.abs(np.mean(roi_gray) - np.mean(bg_roi_gray)) / (np.mean(roi_gray) + np.mean(bg_roi_gray))) + (np.abs(np.mean(roi) - np.mean(bg_roi)) / (np.mean(roi) + np.mean(bg_roi)))) / 2
    return cr

class Evaluate():
    def __init__(self, cfgs, save_path, eval_info):
        self.cfgs = cfgs
        self.datatype = cfgs['datatype']
        self.save_path = save_path
        self.eval_info = eval_info
        self.eval_data = {}
        self.sv = 1540 # [m/s]
        self.mf = 7.8 # [MHz]
        self.sample = 0.5
        self.resolution = self.sample * self.sv / self.mf /1000 # [mm]
        for dataset_id in ['0001', '0002', '0003', '0004', '0009', '0010']:
            self.eval_data[dataset_id] = {}
            for metric in ['input_lateral_fwhm', 'gt_lateral_fwhm', 'pred_lateral_fwhm', 'input_axial_fwhm', 'gt_axial_fwhm', 'pred_axial_fwhm']:
                self.eval_data[dataset_id][metric] = []
        for dataset_id in ['0005', '0006', '0007', '0008']:
            self.eval_data[dataset_id] = {}
            for datatype in ['input', 'gt', 'pred']:
                for metric in ['bg_snr', 'snr', 'cr_image', 'cr_envelope', 'cnr_image', 'cnr_envelope', 'gcnr','cr_prop']:
                    self.eval_data[dataset_id][datatype + '_' + metric] = []

    def main(self, image_path, input, gt, pred, raw_data, phantom):
        dataset_id = image_path.split('/')[-2]
        self.image_path, self.input, self.gt, self.pred, self.raw_data = image_path, input, gt, pred, raw_data
        self.input_image, self.input_envelope = self.transform2image(input, raw_data, self.datatype)
        self.gt_image, self.gt_envelope = self.transform2image(gt, raw_data, self.datatype)
        if self.cfgs['model'] == 'Li':
            self.pred_image, self.pred_envelope = self.transform2image(torch.squeeze(pred), raw_data, 'rf_real')
        else:
            self.pred_image, self.pred_envelope = self.transform2image(pred, raw_data, self.datatype)
        self.save_image()
        self.save_envelope_hist()
        
        if phantom == 'qap':
            coord_info = self.eval_info[os.path.join('/', image_path.split('/')[-4], image_path.split('/')[-3], image_path.split('/')[-2], image_path.split('/')[-1])]
            if dataset_id in ['0001', '0002', '0003', '0004', '0009', '0010']:
                # self.save_plot_image(coord_info[0])
                input_lateral_fwhm, gt_lateral_fwhm, pred_lateral_fwhm, input_axial_fwhm, gt_axial_fwhm, pred_axial_fwhm = self.plot_spreadfunc(coord_info[0])
                for metric, fwhm in zip(['input_lateral_fwhm', 'gt_lateral_fwhm', 'pred_lateral_fwhm', 'input_axial_fwhm', 'gt_axial_fwhm', 'pred_axial_fwhm'], [input_lateral_fwhm, gt_lateral_fwhm, pred_lateral_fwhm, input_axial_fwhm, gt_axial_fwhm, pred_axial_fwhm]):
                    self.eval_data[dataset_id][metric].append(np.array(fwhm))

            if dataset_id in ['0005', '0006', '0007', '0008']:
                input_metric, gt_metric, pred_metric = self.plot_circle(coord_info)
                for metric, datatype in zip([input_metric, gt_metric, pred_metric], ['input', 'gt', 'pred']):
                    for metric_name in ['bg_snr', 'snr', 'cr_image', 'cr_envelope', 'cnr_image', 'cnr_envelope', 'gcnr', 'cr_prop']:
                        self.eval_data[dataset_id][datatype + '_' + metric_name].append(metric[metric_name])

    def transform2image(self, input, raw_data, datatype):
        if datatype in ['rf_real', 'rf_imag']:
            input = input * torch.max(torch.abs(raw_data))
            envelope = torch.abs(input) / torch.max(torch.abs(input))
            input = 20 * torch.log10(envelope)
            input[input < -60] = -60
            input = input + torch.abs(input.min())
            input = input / 60
            X, Y = np.meshgrid(np.arange(input.shape[-1]), np.linspace(0, input.shape[-2]-1, int(input.shape[-2]/2)))
            input = torch.from_numpy(cv2.remap(input.numpy(), X.astype('float32'), Y.astype('float32'), cv2.INTER_CUBIC))
            input = input + torch.abs(input.min())
            envelope = torch.from_numpy(cv2.remap(envelope.numpy(), X.astype('float32'), Y.astype('float32'), cv2.INTER_CUBIC))
            envelope = envelope + torch.abs(envelope.min())
        elif datatype == 'rf':
            real, imag = input[0, :, :] * torch.max(torch.abs(raw_data[0, :, :])), input[1, :, :] * torch.max(torch.abs(raw_data[1, :, :]))
            rf = torch.complex(real, imag)
            envelope = torch.abs(rf) / torch.max(torch.abs(rf))
            input = 20 * torch.log10(envelope)
            input[input < -60] = -60
            input = input / 60
            X, Y = np.meshgrid(np.arange(input.shape[-1]), np.linspace(0, input.shape[-2]-1, int(input.shape[-2]/2)))
            input = torch.from_numpy(cv2.remap(input.numpy(), X.astype('float32'), Y.astype('float32'), cv2.INTER_CUBIC))
            input = input + torch.abs(input.min())
            envelope = torch.from_numpy(cv2.remap(envelope.numpy(), X.astype('float32'), Y.astype('float32'), cv2.INTER_CUBIC))
            envelope = envelope + torch.abs(envelope.min())
        return input/input.max(), envelope/envelope.max()


    def save_image(self):
        os.makedirs(os.path.join(self.save_path, 'Image', self.image_path.split('/')[-4] + self.image_path.split('/')[-2]), exist_ok=True)
        for datatype, image in zip(['input', 'gt', 'pred'], [self.input_image, self.gt_image, self.pred_image]):
            image = Image.fromarray((image.numpy() * 255).astype(np.uint8))
            image.save(os.path.join(self.save_path, 'Image', self.image_path.split('/')[-4] + self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '_{}'.format(datatype) + '.png'))
        fig = plt.figure(tight_layout=True)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.imshow(self.gt_image[:, :], 'gray');ax1.axis('off');ax1.set_title('Compound Image')
        ax2.imshow(self.input_image[:, :], 'gray');ax2.axis('off');ax2.set_title('Input Image')
        ax3.imshow(self.pred_image[:, :], 'gray');ax3.axis('off');ax3.set_title('Estimated Image')
        plt.savefig(os.path.join(self.save_path, 'Image', self.image_path.split('/')[-4] + self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.png'))
        plt.clf()
        plt.close()

    def save_envelope_hist(self):
        os.makedirs(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-4] + self.image_path.split('/')[-2]), exist_ok=True)
        x = np.linspace(0,0.5, 256)
        plt.hist(self.input_envelope.flatten(),bins=x,alpha=0.5, density=True, color='r', label= 'input')
        plt.hist(self.gt_envelope.flatten(),bins=x,alpha=0.5, density=True, color='darkorange', label= 'gt')
        plt.hist(self.pred_envelope.flatten(),bins=x,alpha=0.5, density=True, color='steelblue', label= 'pred')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-4] + self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.png'))
        plt.savefig(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-4] + self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.eps'))
        plt.clf()
        plt.close()

    #annotation plot_lateral
    def save_plot_image(self, coord_info):
        os.makedirs(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2]), exist_ok=True)
        fig, ax = plt.subplots()
        ax.imshow(self.gt_image, 'gray')
        ax.axis('off')
        for i, coord in enumerate(coord_info):
            coord[0]  = coord[0]
            coord[1]  = coord[1]
            ax.plot(coord[0], coord[1], 'r.')
            ax.text(coord[0], coord[1] - 5, str(i+1), color='r')
       
        plt.savefig(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.png'))
        plt.savefig(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.eps'))
        plt.clf()
        plt.close()
    
    def save_roi_hist(self, roi, bg, num, datatype, dataform):
        if dataform == 'envelope':
            x = np.linspace(0,1, 256)
            plt.hist(roi,bins=x,alpha=0.8, density=True, color='darkorange', label= 'roi')
            plt.hist(bg,bins=x,alpha=0.8, density=True, color='steelblue', label='bg')
            plt.legend()
            plt.savefig(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-2], self.image_path.split('/')[-1],'Circle{0:04d}'.format(num+1) + '_{}'.format(datatype) + '_{}'.format(dataform) + '.png'))
            plt.savefig(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-2], self.image_path.split('/')[-1],'Circle{0:04d}'.format(num+1) + '_{}'.format(datatype) + '_{}'.format(dataform) + '.eps'))
        if dataform == 'gray':
            x = np.linspace(0,255, 256)
            plt.hist(roi,bins=x,alpha=0.8, color='darkorange', label= 'roi')
            plt.hist(bg,bins=x,alpha=0.8, color='steelblue', label='bg')
            plt.legend()
            plt.savefig(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-2], self.image_path.split('/')[-1],'Circle{0:04d}'.format(num+1) + '_{}'.format(datatype) + '_{}'.format(dataform) + '.png'))
            plt.savefig(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-2], self.image_path.split('/')[-1],'Circle{0:04d}'.format(num+1) + '_{}'.format(datatype) + '_{}'.format(dataform) + '.eps'))
        plt.clf()
        plt.close()

    def plot_circle(self, coord_info):
        # os.makedirs(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2]), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'Metric', self.image_path.split('/')[-2]), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-2], self.image_path.split('/')[-1]), exist_ok=True)
        # fig, ax = plt.subplots()
        # ax.imshow(self.gt_image, 'gray')
        # ax.axis('off')
        # theta = np.linspace(0, 2*np.pi, 100)
        # for i, coords in enumerate(coord_info[0]):
        #     x_coord, y_coord, radius = coords[0], coords[1], coords[2]
        #     mask_out_1 = np.zeros_like(self.gt_image)
        #     mask_out_2 = np.zeros_like(self.gt_image)
        #     cv2.circle(mask_out_1, center = (round(x_coord), round(y_coord)), radius = radius+5, color=1, thickness=-1)
        #     cv2.circle(mask_out_2, center = (round(x_coord), round(y_coord)), radius = int(np.sqrt(pow(radius-5,2)+pow(radius+5,2))), color=1, thickness=-1)
        #     mask = cv2.bitwise_xor(mask_out_1, mask_out_2)
        #     mask = np.where(mask==1)
        #     ax.text(x_coord, y_coord - radius - 25, 'Circle{}'.format(i+1), color='r')
        #     ax.plot(x_coord + radius * np.cos(theta), y_coord + radius * np.sin(theta), 'r')
        #     ax.plot(x_coord + (radius + 5) * np.cos(theta), y_coord + (radius + 5) * np.sin(theta), 'steelblue')
        #     ax.plot(x_coord + np.sqrt(pow(radius-5,2)+pow(radius+5,2)) * np.cos(theta), y_coord + np.sqrt(pow(radius-5,2)+pow(radius+5,2)) * np.sin(theta), 'steelblue')
        #     ax.scatter(x=mask[1], y=mask[0], c='steelblue', s=0.5, alpha=0.1)
        #     ax.plot(x_coord + (radius - 5) * np.cos(theta), y_coord + (radius - 5) * np.sin(theta), color='darkorange')
        #     ax.fill(x_coord + (radius - 5) * np.cos(theta), y_coord + (radius - 5) * np.sin(theta),facecolor='darkorange', alpha=0.3)
        # plt.savefig(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.png'))
        # plt.savefig(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.eps'))
        # plt.clf()
        # plt.close()

        for i, coords in enumerate(coord_info[0]):
            gt_metric = self.compute_circle(self.gt_image, self.gt_envelope, coord_info, 'gt')
            input_metric = self.compute_circle(self.input_image, self.input_envelope, coord_info, 'input')
            pred_metric = self.compute_circle(self.pred_image,self.pred_envelope, coord_info, 'pred')

        df = pd.DataFrame([], columns=['Datatype', 'BG'] + gt_metric['circle'])
        for metric, metric_name in zip(['snr', 'cr_image', 'cr_envelope', 'cnr_image', 'cnr_envelope', 'gcnr', 'cr_prop'], ['SNR', 'CR_IMG', 'CR_ENV', 'CNR_IMG', 'CNR_ENV', 'GCNR', 'CR_Prop']):
            df = df.append(pd.Series([metric_name] + [''] * (1 + len(gt_metric['circle'])), index=df.columns), ignore_index=True)
            if metric == 'snr':
                df = df.append(pd.Series(['Input'] + input_metric['bg_snr'] + input_metric[metric], index=df.columns), ignore_index=True)
                df = df.append(pd.Series(['GT'] + gt_metric['bg_snr'] + gt_metric[metric], index=df.columns), ignore_index=True)
                df = df.append(pd.Series(['Pred'] + pred_metric['bg_snr'] + pred_metric[metric], index=df.columns), ignore_index=True)
            else:
                df = df.append(pd.Series(['Input'] + [''] + input_metric[metric], index=df.columns), ignore_index=True)
                df = df.append(pd.Series(['GT'] + [''] + gt_metric[metric], index=df.columns), ignore_index=True)
                df = df.append(pd.Series(['Pred'] + [''] + pred_metric[metric], index=df.columns), ignore_index=True)
        df.to_csv(os.path.join(self.save_path, 'Metric', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.csv'))
        return input_metric, gt_metric, pred_metric

    def compute_circle(self, image, envelope, coord_info, datatype):
        bg_snr_mean_list, bg_snr_list, snr_list, cr_image_list, cr_envelope_list, cnr_image_list, cnr_envelope_list, gcnr_list, cr_prop_list, circle_list = [], [], [], [], [], [], [], [], [], []
        for i, coords in enumerate(coord_info[0]):
            bg_roi_image, roi_image = self.roi_circle(image, coords[0], coords[1], coords[2])
            bg_roi_image, roi_image = np.round(bg_roi_image * 255), np.round(roi_image * 255)
            bg_roi_envelope, roi_envelope = self.roi_circle(envelope, coords[0], coords[1], coords[2])
            bg_snr_list.append(float(signal_to_noise(bg_roi_image)))
            snr_list.append(float(signal_to_noise(roi_image)))
            cr_image_list.append(float(contrast_gray(roi_image, bg_roi_image)))
            cr_envelope_list.append(float(contrast_envelope(roi_envelope, bg_roi_envelope)))
            cnr_image_list.append(float(contrast_to_noise_gray(roi_image, bg_roi_image)))
            cnr_envelope_list.append(float(contrast_to_noise_envelope(roi_envelope, bg_roi_envelope)))
            gcnr_list.append(float(generalized_contrast_to_noise(roi_envelope, bg_roi_envelope)))
            cr_prop_list.append(float(cr_prop_fn(roi_envelope, roi_image, bg_roi_envelope, bg_roi_image)))
            circle_list.append('Circle{0:04d}'.format(i+1))
            self.save_roi_hist(roi_image, bg_roi_image, i, datatype, 'gray')
            self.save_roi_hist(roi_envelope, bg_roi_envelope, i, datatype, 'envelope')
        bg_snr_mean_list = [sum(bg_snr_list)/len(bg_snr_list)]
        metric = dict(bg_snr=bg_snr_mean_list, snr=snr_list, cr_image=cr_image_list, cr_envelope=cr_envelope_list, cnr_image=cnr_image_list, cnr_envelope=cnr_envelope_list, gcnr=gcnr_list, cr_prop=cr_prop_list, circle=circle_list)
        return metric

    def roi_circle(self, input, x, y, r):
        bg_mask_in = np.zeros_like(input)
        bg_mask_out = np.zeros_like(input)
        cv2.circle(bg_mask_in, center = (round(x),round(y)), radius = r+5, color=1, thickness=-1)
        cv2.circle(bg_mask_out, center = (round(x),round(y)), radius = int(np.sqrt(pow(r-5,2)+pow(r+5,2))), color=1, thickness=-1)
        bg_mask = cv2.bitwise_xor(bg_mask_in,bg_mask_out)
        bg_roi = bg_mask * input.numpy()
        bg_roi_data = bg_roi.flatten()[np.nonzero(bg_roi.flatten())]
        roi_mask_in = np.zeros_like(input)
        roi_mask_out = np.zeros_like(input)
        cv2.circle(roi_mask_out, center = (round(x),round(y)), radius = r-5, color=1, thickness=-1)
        roi_mask = cv2.bitwise_xor(roi_mask_in,roi_mask_out)
        roi = roi_mask * input.numpy()
        roi_data = roi.flatten()[np.nonzero(roi.flatten())]
        return bg_roi_data, roi_data

    def visualize_rf_spectrum(self):
        os.makedirs(os.path.join(self.save_path, 'RealRFSpectrum', self.image_path.split('/')[-2]), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'ImagRFSpectrum', self.image_path.split('/')[-2]), exist_ok=True)

        if self.datatype == 'rf':
            spectrum_type = ['RealRFSpectrum', 'ImagRFSpectrum']
        elif self.datatype == 'rf_real':
            spectrum_type = ['RealRFSpectrum']
        elif self.datatype == 'rf_imag':
            spectrum_type = ['ImagRFSpectrum']

        spectrum_list = []
        for datatype, rf in zip(['input', 'gt', 'pred'], [self.input, self.gt, self.pred]):
            fft = np.fft.fft(rf, axis=2)
            shift_fft = np.fft.fftshift(fft, axes=2)
            spectrum = 20 * np.log(np.abs(shift_fft))
            spectrum_list.append(spectrum)
            for i in range(spectrum.shape[0]):
                plt.figure()
                plt.imshow(spectrum[i])
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_path, spectrum_type[i], self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '_{}'.format(datatype) + '.png'))
                plt.clf()
                plt.close()

        for i in range(spectrum.shape[0]):
            fig = plt.figure(tight_layout=True)
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            ax1.imshow(spectrum_list[0][i, :, :]);ax1.axis('off');ax1.set_title('Compound Image')
            ax2.imshow(spectrum_list[1][i, :, :]);ax2.axis('off');ax2.set_title('Input Image')
            ax3.imshow(spectrum_list[2][i, :, :]);ax3.axis('off');ax3.set_title('Estimated Image')
            plt.savefig(os.path.join(self.save_path, spectrum_type[i], self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.png'))
            plt.clf()
            plt.close()

    def visualize_image_spectrum(self):
        os.makedirs(os.path.join(self.save_path, 'ImageSpectrum', self.image_path.split('/')[-2]), exist_ok=True)
        spectrum_list = []
        for datatype, image in zip(['input', 'gt', 'pred'], [self.input_image, self.gt_image, self.pred_image]):
            fft = np.fft.fft2(image)
            shift_fft = np.fft.fftshift(fft)
            spectrum = 20 * np.log(np.abs(shift_fft))
            spectrum_list.append(spectrum)
            plt.figure()
            plt.imshow(spectrum)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, 'ImageSpectrum', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '_{}'.format(datatype) + '.png'))
            plt.clf()
            plt.close()
        fig = plt.figure(tight_layout=True)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.imshow(spectrum_list[0][:, :]);ax1.axis('off');ax1.set_title('Compound Image')
        ax2.imshow(spectrum_list[1][:, :]);ax2.axis('off');ax2.set_title('Input Image')
        ax3.imshow(spectrum_list[2][:, :]);ax3.axis('off');ax3.set_title('Estimated Image')
        plt.savefig(os.path.join(self.save_path, 'ImageSpectrum', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.png'))
        plt.clf()
        plt.close()

    def plot_spreadfunc(self, eval_info):
        os.makedirs(os.path.join(self.save_path, 'SpreadFunc', self.image_path.split('/')[-2],'Lateral'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'SpreadFunc', self.image_path.split('/')[-2],'Axial'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'Metric', self.image_path.split('/')[-2]), exist_ok=True)
        line, thr, point_num, rf_list = 8, 40, 50, []
        for rf in [self.input, self.gt, self.pred]:
            real, imag = rf[0, :, :] * torch.max(torch.abs(self.raw_data[0, :, :])), rf[1, :, :] * torch.max(torch.abs(self.raw_data[1, :, :]))
            rf = torch.complex(real, imag)
            envelope = torch.abs(rf) / torch.max(torch.abs(rf))
            envelope = pixel_remap(envelope)
            rf_list.append(envelope)

        input_lateral_fwhm_list, gt_lateral_fwhm_list, pred_lateral_fwhm_list,input_axial_fwhm_list, gt_axial_fwhm_list, pred_axial_fwhm_list, point_list = [], [], [], [], [], [], []
        for i, coord in enumerate(eval_info):
        # lateral
            input_signal_lateral = rf_list[0][coord[1], max(0, coord[0] - line):coord[0] + line]
            gt_signal_lateral = rf_list[1][coord[1], max(0, coord[0] - line):coord[0] + line]
            pred_signal_lateral = rf_list[2][coord[1], max(0, coord[0] - line):coord[0] + line]
            plot_coord_lateral = coord[0]
            signal_shape_lateral = input_signal_lateral.shape[0]
            
            f = interp1d(np.linspace(0, signal_shape_lateral - 1, num=input_signal_lateral.shape[0]), input_signal_lateral)
            input_signal_lateral = torch.tensor(f(np.linspace(0, signal_shape_lateral - 1, num=point_num)))
            f = interp1d(np.linspace(0, signal_shape_lateral - 1, num=gt_signal_lateral.shape[0]), gt_signal_lateral)
            gt_signal_lateral = torch.tensor(f(np.linspace(0, signal_shape_lateral - 1, num=point_num)))
            f = interp1d(np.linspace(0, signal_shape_lateral - 1, num=pred_signal_lateral.shape[0]), pred_signal_lateral)
            pred_signal_lateral = torch.tensor(f(np.linspace(0, signal_shape_lateral - 1, num=point_num)))

            # FWHM
            input_lateral_fwhm_list.append(self.compute_FWHM(input_signal_lateral) * signal_shape_lateral / point_num)
            gt_lateral_fwhm_list.append(self.compute_FWHM(gt_signal_lateral) * signal_shape_lateral / point_num)
            pred_lateral_fwhm_list.append(self.compute_FWHM(pred_signal_lateral) * signal_shape_lateral / point_num)
            
            # dB transform
            input_signal_lateral = 20 * torch.log10(input_signal_lateral)
            input_signal_lateral[input_signal_lateral < -thr] = -thr
            gt_signal_lateral = 20 * torch.log10(gt_signal_lateral)
            gt_signal_lateral[gt_signal_lateral < -thr] = -thr
            pred_signal_lateral = 20 * torch.log10(pred_signal_lateral)
            pred_signal_lateral[pred_signal_lateral < -thr] = -thr
            
            # Visualize
            plt.figure()
            plt.plot(np.linspace(max(0, plot_coord_lateral - line), min(rf_list[0].shape[0], plot_coord_lateral + line), num=point_num), input_signal_lateral, marker='.', ls=':', color='b', label='Input')
            plt.plot(np.linspace(max(0, plot_coord_lateral - line), min(rf_list[0].shape[0], plot_coord_lateral + line), num=point_num), gt_signal_lateral, marker='.', ls=':', color='g', label='GT')
            plt.plot(np.linspace(max(0, plot_coord_lateral - line), min(rf_list[0].shape[0], plot_coord_lateral + line), num=point_num), pred_signal_lateral, marker='.', ls=':', color='r', label='Pred')
            plt.tight_layout()
            plt.ylim([-thr-5, 0])
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(self.save_path, 'SpreadFunc', self.image_path.split('/')[-2], 'Lateral', self.image_path.split('/')[-1] + '_{0:04}.png'.format(i+1)))
            plt.savefig(os.path.join(self.save_path, 'SpreadFunc', self.image_path.split('/')[-2], 'Lateral', self.image_path.split('/')[-1] + '_{0:04}.eps'.format(i+1)))
            plt.clf()
            plt.close()

        # axial
            input_signal_axial = rf_list[0][max(0, coord[1] - line):min(rf_list[0].shape[0], coord[1] + line), coord[0]]
            gt_signal_axial = rf_list[1][max(0, coord[1] - line):min(rf_list[1].shape[0], coord[1] + line), coord[0]]
            pred_signal_axial = rf_list[2][max(0, coord[1] - line):min(rf_list[2].shape[0], coord[1] + line), coord[0]]
            plot_coord_axial = coord[1]
            signal_shape_axial = input_signal_axial.shape[0]

            f = interp1d(np.linspace(0, signal_shape_axial - 1, num=input_signal_axial.shape[0]), input_signal_axial)
            input_signal_axial = torch.tensor(f(np.linspace(0, signal_shape_axial - 1, num=point_num)))
            f = interp1d(np.linspace(0, signal_shape_axial - 1, num=gt_signal_axial.shape[0]), gt_signal_axial)
            gt_signal_axial = torch.tensor(f(np.linspace(0, signal_shape_axial - 1, num=point_num)))
            f = interp1d(np.linspace(0, signal_shape_axial - 1, num=pred_signal_axial.shape[0]), pred_signal_axial)
            pred_signal_axial = torch.tensor(f(np.linspace(0, signal_shape_axial - 1, num=point_num)))

            # FWHM
            input_axial_fwhm_list.append(self.compute_FWHM(input_signal_axial) * signal_shape_axial / point_num)
            gt_axial_fwhm_list.append(self.compute_FWHM(gt_signal_axial) * signal_shape_axial / point_num)
            pred_axial_fwhm_list.append(self.compute_FWHM(pred_signal_axial) * signal_shape_axial / point_num)

            # dB transform
            input_signal_axial = 20 * torch.log10(input_signal_axial)
            input_signal_axial[input_signal_axial < -thr] = -thr
            gt_signal_axial = 20 * torch.log10(gt_signal_axial)
            gt_signal_axial[gt_signal_axial < -thr] = -thr
            pred_signal_axial = 20 * torch.log10(pred_signal_axial)
            pred_signal_axial[pred_signal_axial < -thr] = -thr

            # Visualize
            plt.figure()
            plt.plot(np.linspace(max(0, plot_coord_axial - line), min(rf_list[0].shape[0], plot_coord_axial + line), num=point_num), input_signal_axial, marker='.', ls=':', color='b', label='Input')
            plt.plot(np.linspace(max(0, plot_coord_axial - line), min(rf_list[0].shape[0], plot_coord_axial + line), num=point_num), gt_signal_axial, marker='.', ls=':', color='g', label='GT')
            plt.plot(np.linspace(max(0, plot_coord_axial - line), min(rf_list[0].shape[0], plot_coord_axial + line), num=point_num), pred_signal_axial, marker='.', ls=':', color='r', label='Pred')
            plt.tight_layout()
            plt.ylim([-thr-5, 0])
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(self.save_path, 'SpreadFunc', self.image_path.split('/')[-2], 'Axial', self.image_path.split('/')[-1] + '_{0:04}.png'.format(i+1)))
            plt.savefig(os.path.join(self.save_path, 'SpreadFunc', self.image_path.split('/')[-2], 'Axial', self.image_path.split('/')[-1] + '_{0:04}.eps'.format(i+1)))
            plt.clf()
            plt.close()
            
            point_list.append('Point{0:04d}'.format(i+1))
            
        df = pd.DataFrame([], columns=['Datatype'] + point_list)
        df = df.append(pd.Series(['Lateral'] + ['']*len(gt_lateral_fwhm_list), index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['GT'] + gt_lateral_fwhm_list, index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['Input'] + input_lateral_fwhm_list, index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['Pred'] + pred_lateral_fwhm_list, index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['Axial'] + ['']*len(gt_axial_fwhm_list), index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['GT'] + gt_axial_fwhm_list, index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['Input'] + input_axial_fwhm_list, index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['Pred'] + pred_axial_fwhm_list, index=df.columns), ignore_index=True)
        df.to_csv(os.path.join(self.save_path, 'Metric', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.csv'))
        
        return input_lateral_fwhm_list, gt_lateral_fwhm_list, pred_lateral_fwhm_list, input_axial_fwhm_list, gt_axial_fwhm_list, pred_axial_fwhm_list


    def compute_FWHM(self, signal):
        line = signal.shape[0] // 2
        mask = np.nonzero(signal >= (torch.max(signal) * 0.5))
        max_index = torch.where(signal == torch.max(signal))[0][0].item()
        mask_index = mask[mask == max_index].item()
        index = mask_index - 1
        for i in range(1, line):
            if mask[mask == index].shape[0] == 0 or index == 0:
                break
            index = index - 1
        fwhm_min_index = index
        index = mask_index + 1
        for i in range(1, line):
            if mask[mask == index].shape[0] == 0 or index == signal.shape[0] - 1:
                break                    
            index = index + 1
        fwhm_max_index = index
        if (fwhm_min_index == 0 and mask[0] == 0) or (fwhm_max_index == signal.shape[0] - 1 and mask[-1] == signal.shape[0] - 1):
            fwhm = 0
        else:
            fwhm = fwhm_max_index - fwhm_min_index
        return fwhm * self.resolution
    
    def summary_result(self):
        for dataset_id in ['0001', '0002', '0003', '0004', '0009', '0010']:
            df = pd.DataFrame([], columns=['Datatype_Lateral', 'All', 'Num. of FWHM', 'Per. of FWHM', 'Used Num. of FWHM', 'Avg_FWHM'])
            input_lateral_fwhm =  np.concatenate(self.eval_data[dataset_id]['input_lateral_fwhm'])
            gt_lateral_fwhm = np.concatenate(self.eval_data[dataset_id]['gt_lateral_fwhm'])
            pred_lateral_fwhm = np.concatenate(self.eval_data[dataset_id]['pred_lateral_fwhm'])       
            idx_lateral = np.logical_and(input_lateral_fwhm, gt_lateral_fwhm)
            idx_lateral = np.logical_and(idx_lateral, pred_lateral_fwhm)
            for metric, datatype in zip(['gt_lateral_fwhm', 'input_lateral_fwhm', 'pred_lateral_fwhm'], ['GT', 'Input', 'Pred']):
                self.eval_data[dataset_id][metric] = np.array([e for inner_list in self.eval_data[dataset_id][metric] for e in inner_list])
                all_count = self.eval_data[dataset_id][metric].shape[0]
                fwhm_count = (self.eval_data[dataset_id][metric] != 0).sum()
                used_count = np.count_nonzero(idx_lateral)
                avg_fwhm = np.mean(self.eval_data[dataset_id][metric][idx_lateral])
                df = df.append(pd.Series([datatype, all_count, fwhm_count, fwhm_count / all_count * 100, used_count, avg_fwhm], index=df.columns), ignore_index=True)
            df.to_csv(os.path.join(self.save_path, dataset_id + '_Lateral.csv'))
            if dataset_id == '0001':
                lateral_result = np.array(df)
            else:
                lateral_result += np.array(df)

            df = pd.DataFrame([], columns=['Datatype_Axial', 'All', 'Num. of FWHM', 'Per. of FWHM', 'Used Num. of FWHM', 'Avg_FWHM'])
            input_axial_fwhm =  np.concatenate(self.eval_data[dataset_id]['input_axial_fwhm'])
            gt_axial_fwhm = np.concatenate(self.eval_data[dataset_id]['gt_axial_fwhm'])
            pred_axial_fwhm = np.concatenate(self.eval_data[dataset_id]['pred_axial_fwhm'])
            idx_axial = np.logical_and(input_axial_fwhm, gt_axial_fwhm)
            idx_axial = np.logical_and(idx_axial, pred_axial_fwhm)       
            for metric, datatype in zip(['gt_axial_fwhm', 'input_axial_fwhm', 'pred_axial_fwhm'], ['GT', 'Input', 'Pred']):
                self.eval_data[dataset_id][metric] = np.array([e for inner_list in self.eval_data[dataset_id][metric] for e in inner_list])
                all_count = self.eval_data[dataset_id][metric].shape[0]
                fwhm_count = (self.eval_data[dataset_id][metric] != 0).sum()
                used_count = np.count_nonzero(idx_axial)
                avg_fwhm = np.mean(self.eval_data[dataset_id][metric][idx_axial])
                df = df.append(pd.Series([datatype, all_count, fwhm_count, fwhm_count / all_count * 100, used_count, avg_fwhm], index=df.columns), ignore_index=True)
            df.to_csv(os.path.join(self.save_path, dataset_id + '_Axial.csv'))
            if dataset_id == '0001':
                axial_result = np.array(df)
            else:
                axial_result += np.array(df)

        lateral_result = np.delete(lateral_result, 0, axis=1)
        lateral_result = np.delete(lateral_result, 2, axis=1)
        lateral_result[:, -1] = lateral_result[:, -1]/6
        df = pd.DataFrame(lateral_result, columns=['All', 'Num. of FWHM', 'Used Num. of FWHM', 'Avg_FWHM'])
        df.insert(0, 'Datatype_Lateral', ['GT', 'Input', 'Pred'])
        df.to_csv(os.path.join(self.save_path, 'Result_Lateral.csv'))
        
        axial_result = np.delete(axial_result, 0, axis=1)
        axial_result = np.delete(axial_result, 2, axis=1)
        axial_result[:, -1] = axial_result[:, -1]/6
        df = pd.DataFrame(axial_result, columns=['All', 'Num. of FWHM', 'Used Num. of FWHM', 'Avg_FWHM'])
        df.insert(0, 'Datatype_Axial', ['GT', 'Input', 'Pred'])
        df.to_csv(os.path.join(self.save_path, 'Result_Axail.csv'))

        for dataset_id in ['0005', '0006', '0007', '0008']:
            df = pd.DataFrame([], columns=['Datatype', 'BG_SNR', 'SNR', 'CR_IMG', 'CR_ENV', 'CNR_IMG', 'CNR_ENV', 'GCNR', 'CR_Prop'])
            for datatype_name, datatype in zip(['GT', 'Input', 'Pred'], ['gt', 'input', 'pred']):
                metric_list = []
                for metric, metric_name in zip(['bg_snr', 'snr', 'cr_image', 'cr_envelope', 'cnr_image', 'cnr_envelope', 'gcnr', 'cr_prop'], ['BG_SNR', 'SNR', 'CR_IMG', 'CR_ENV', 'CNR_IMG', 'CNR_ENV', 'GCNR', 'CR_Prop']):
                    self.eval_data[dataset_id][datatype + '_' + metric] = np.array([e for inner_list in self.eval_data[dataset_id][datatype + '_' + metric] for e in inner_list])
                    metric_list.append(np.mean(self.eval_data[dataset_id][datatype + '_' + metric]))
                df = df.append(pd.Series([datatype_name] + metric_list, index=df.columns), ignore_index=True)
            df.to_csv(os.path.join(self.save_path, dataset_id + '.csv'))
            if dataset_id == '0005':
                contrast_p_result = np.array(df)
            elif dataset_id == '0006':
                contrast_p_result += np.array(df)
            elif dataset_id == '0007':
                contrast_n_result = np.array(df)
            elif dataset_id == '0008':
                contrast_n_result += np.array(df)
        
        contrast_p_result = np.delete(contrast_p_result, 0, axis=1)
        contrast_p_result = contrast_p_result / 2
        df = pd.DataFrame(contrast_p_result, columns=['BG_SNR', 'SNR', 'CR_IMG', 'CR_ENV', 'CNR_IMG', 'CNR_ENV', 'GCNR', 'CR_Prop'])
        df.insert(0, 'Datatype', ['GT', 'Input', 'Pred'])
        df.to_csv(os.path.join(self.save_path, 'Result_Contrast_Positive.csv'))

        contrast_n_result = np.delete(contrast_n_result, 0, axis=1)
        contrast_n_result = contrast_n_result / 2
        df = pd.DataFrame(contrast_n_result, columns=['BG_SNR', 'SNR', 'CR_IMG', 'CR_ENV', 'CNR_IMG', 'CNR_ENV', 'GCNR', 'CR_Prop'])
        df.insert(0, 'Datatype', ['GT', 'Input', 'Pred'])
        df.to_csv(os.path.join(self.save_path, 'Result_Contrast_Negative.csv'))


class PICMUS_Evaluate():
    def __init__(self, cfgs, save_path):
        self.PICMUS_hdf5_path = cfgs['PICMUS_hdf5_path']
        self.cfgs = cfgs
        self.datatype = cfgs['datatype']
        self.save_path = save_path
        self.eval_data = {}
        self.eps = 1e-10
        for dataset_id in ['resolution_distorsion_expe', 'resolution_distorsion_simu']:
            self.eval_data[dataset_id] = {}
            for metric in ['input_lateral_fwhm', 'gt_lateral_fwhm', 'pred_lateral_fwhm', 'input_axial_fwhm', 'gt_axial_fwhm', 'pred_axial_fwhm']:
                self.eval_data[dataset_id][metric] = []
        for dataset_id in ['contrast_speckle_expe', 'contrast_speckle_simu']:
            self.eval_data[dataset_id] = {}
            for datatype in ['input', 'gt', 'pred']:
                for metric in ['bg_snr', 'snr', 'cr_image', 'cr_envelope', 'cnr_image', 'cnr_envelope', 'gcnr', 'cr_prop']:
                    self.eval_data[dataset_id][datatype + '_' + metric] = []

    def main(self, image_path, input, gt, pred, raw_data):
        dataset_id = image_path.split('/')[-1]
        self.image_path, self.input, self.gt, self.pred, self.raw_data = image_path, input, gt, pred, raw_data
        self.input_image, self.input_envelope = self.transform2image(input, raw_data, self.datatype)
        self.gt_image, self.gt_envelope = self.transform2image(gt, raw_data, self.datatype)
        if self.cfgs['model'] == 'Li':
            self.pred_image, self.pred_envelope = self.transform2image(torch.squeeze(pred), raw_data, 'rf_real')
        else:
            self.pred_image, self.pred_envelope = self.transform2image(pred, raw_data, self.datatype)
        self.save_image()
        if dataset_id in ['resolution_distorsion_expe', 'resolution_distorsion_simu']:
            coord_info, self.resolution = self.get_coord_info(dataset_id)
            # self.save_plot_image(coord_info[0])
            input_lateral_fwhm, gt_lateral_fwhm, pred_lateral_fwhm, input_axial_fwhm, gt_axial_fwhm, pred_axial_fwhm = self.plot_spreadfunc(coord_info[0])
            for metric, fwhm in zip(['input_lateral_fwhm', 'gt_lateral_fwhm', 'pred_lateral_fwhm', 'input_axial_fwhm', 'gt_axial_fwhm', 'pred_axial_fwhm'], [input_lateral_fwhm, gt_lateral_fwhm, pred_lateral_fwhm, input_axial_fwhm, gt_axial_fwhm, pred_axial_fwhm]):
                self.eval_data[dataset_id][metric].append(np.array(fwhm))

        if dataset_id in ['contrast_speckle_expe', 'contrast_speckle_simu']:
            coord_info, self.resolution = self.get_coord_info(dataset_id)
            input_metric, gt_metric, pred_metric = self.plot_circle(coord_info)
            for metric, datatype in zip([input_metric, gt_metric, pred_metric], ['input', 'gt', 'pred']):
                for metric_name in ['bg_snr', 'snr', 'cr_image', 'cr_envelope', 'cnr_image', 'cnr_envelope', 'gcnr', 'cr_prop']:
                    self.eval_data[dataset_id][datatype + '_' + metric_name].append(metric[metric_name])

    def make_pixel_grid(self, x_axis, z_axis,xlims, zlims, dx, dz):
        x = np.arange(xlims[0], xlims[1] + self.eps, dx)
        z = np.arange(zlims[0], zlims[1] + self.eps, dz)
        xx, zz = np.meshgrid(np.linspace(0, x_axis.shape[0]-1, int(x.shape[0])), np.linspace(0, z_axis.shape[0]-1, int(z.shape[0])))
        return zz, xx
    
    def transform2image(self, input, raw_data, datatype):
        if self.image_path.split('/')[-1].split('_')[0] == 'contrast':
            if self.image_path.split('/')[-1].split('_')[-1] == 'simu':
                f = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/simulation/contrast_speckle/contrast_speckle_simu_dataset_iq.hdf5'), "r")["US"]["US_DATASET0000"]
                fp = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/simulation/contrast_speckle/contrast_speckle_simu_phantom.hdf5'), "r")["US"]["US_DATASET0000"]
                fs = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/simulation/contrast_speckle/contrast_speckle_simu_scan.hdf5'), "r")["US"]["US_DATASET0000"]
            elif self.image_path.split('/')[-1].split('_')[-1] == 'expe':
                f = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5'), "r")["US"]["US_DATASET0000"]
                fp = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/experiments/contrast_speckle/contrast_speckle_expe_phantom.hdf5'), "r")["US"]["US_DATASET0000"]
                fs = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5'), "r")["US"]["US_DATASET0000"]
        elif self.image_path.split('/')[-1].split('_')[0] == 'resolution':
            if self.image_path.split('/')[-1].split('_')[-1] == 'simu':
                f = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_iq.hdf5'), "r")["US"]["US_DATASET0000"]
                fp = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/simulation/resolution_distorsion/resolution_distorsion_simu_phantom.hdf5'), "r")["US"]["US_DATASET0000"]
                fs = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5'), "r")["US"]["US_DATASET0000"]
            elif self.image_path.split('/')[-1].split('_')[-1] == 'expe':
                f = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/experiments/resolution_distorsion/resolution_distorsion_expe_dataset_iq.hdf5'), "r")["US"]["US_DATASET0000"]
                fp = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/experiments/resolution_distorsion/resolution_distorsion_expe_phantom.hdf5'), "r")["US"]["US_DATASET0000"]
                fs = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/experiments/resolution_distorsion/resolution_distorsion_expe_scan.hdf5'), "r")["US"]["US_DATASET0000"]
        ss = np.array(f["sound_speed"],dtype = "float32")
        mf = np.array(f["modulation_frequency"],dtype = "float32")
        ele_pos = np.array(f["probe_geometry"])
        x_axis = np.array(fs["x_axis"],dtype = "float32")
        z_axis = np.array(fs["z_axis"],dtype = "float32")
        xlims = (ele_pos[0,0] , ele_pos[0,-1])
        zlims = np.array([5e-3, 55e-3])
        wvln = ss / mf
        dx = wvln / 3
        dz = dx  # Use square pixels
        zz, xx = self.make_pixel_grid(x_axis, z_axis, xlims, zlims, dx, dz)
        if datatype == 'us':
            input = input * 0.5 + 0.5
            return input
        elif datatype == 'envelope':
            input = 20 * torch.log10(input)
            input[input < -60] = -60
            input = input + torch.abs(input.min())
            input = input / 60
            return input
        elif datatype in ['rf_real', 'rf_imag']:
            input = input * torch.max(torch.abs(raw_data))
            input = torch.abs(input) / torch.max(torch.abs(input))
            input = 20 * torch.log10(input)
            input[input < -60] = -60
            input = input + torch.abs(input.min())
            input = input / 60
            return input
        elif datatype == 'rf':
            real, imag = input[0, :, :] * torch.max(torch.abs(raw_data[0, :, :])), input[1, :, :] * torch.max(torch.abs(raw_data[1, :, :]))
            rf = torch.complex(real, imag)
            envelope = torch.abs(rf) / torch.max(torch.abs(rf))
            input = 20 * torch.log10(envelope)
            input[input < -60] = -60
            input = input + torch.abs(input.min())
            input = input / 60
            input = torch.from_numpy(cv2.remap(input.numpy(), xx.astype('float32'), zz.astype('float32'), cv2.INTER_CUBIC))
            input = input + torch.abs(input.min())
            envelope = torch.from_numpy(cv2.remap(envelope.numpy(), xx.astype('float32'), zz.astype('float32'), cv2.INTER_CUBIC))
            envelope = envelope + torch.abs(envelope.min())
            return input/input.max(), envelope/envelope.max()

    def save_image(self):
        os.makedirs(os.path.join(self.save_path, 'Image', self.image_path.split('/')[-2]), exist_ok=True)
        for datatype, image in zip(['input', 'gt', 'pred'], [self.input_image, self.gt_image, self.pred_image]):
            image = (image - image.min()) / (image.max() - image.min())
            image = Image.fromarray((image.numpy() * 255).astype(np.uint8))
            image.save(os.path.join(self.save_path, 'Image', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '_{}'.format(datatype) + '.png'))
        fig = plt.figure(tight_layout=True)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.imshow(self.gt_image[:, :], 'gray');ax1.axis('off');ax1.set_title('Compound Image')
        ax2.imshow(self.input_image[:, :], 'gray');ax2.axis('off');ax2.set_title('Input Image')
        ax3.imshow(self.pred_image[:, :], 'gray');ax3.axis('off');ax3.set_title('Estimated Image')
        plt.savefig(os.path.join(self.save_path, 'Image', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.png'))
        plt.clf()
        plt.close()

    def get_coord_info(self, dataset_id):
        temp = np.zeros([int(self.raw_data.shape[-2]),int(self.raw_data.shape[-1])])
        coord_info = []
        if dataset_id.split('_')[0] == 'contrast':
            if dataset_id.split('_')[-1] == 'simu':
                f = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/simulation/contrast_speckle/contrast_speckle_simu_dataset_iq.hdf5'), "r")["US"]["US_DATASET0000"]
                fp = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/simulation/contrast_speckle/contrast_speckle_simu_phantom.hdf5'), "r")["US"]["US_DATASET0000"]
                fs = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/simulation/contrast_speckle/contrast_speckle_simu_scan.hdf5'), "r")["US"]["US_DATASET0000"]
            elif dataset_id.split('_')[-1] == 'expe':
                f = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5'), "r")["US"]["US_DATASET0000"]
                fp = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/experiments/contrast_speckle/contrast_speckle_expe_phantom.hdf5'), "r")["US"]["US_DATASET0000"]
                fs = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5'), "r")["US"]["US_DATASET0000"]
            ss = np.array(f["sound_speed"],dtype = "float32")
            sf = np.array(f["sampling_frequency"],dtype = "float32")
            mf = np.array(f["modulation_frequency"],dtype = "float32")
            ele_pos = np.array(f["probe_geometry"])
            oc_x = np.array(fp["phantom_occlusionCenterX"],dtype = "float32")
            oc_x = np.array(fp["phantom_occlusionCenterX"],dtype = "float32")
            oc_z = np.array(fp["phantom_occlusionCenterZ"],dtype = "float32")
            oc_r =np.array(fp["phantom_occlusionDiameter"],dtype = "float32") / 2 
            x_axis = np.array(fs["x_axis"],dtype = "float32")
            z_axis = np.array(fs["z_axis"],dtype = "float32")
            xlims = (ele_pos[0,0] , ele_pos[0,-1])
            zlims = np.array([5e-3, 55e-3])
            wvln = ss / mf
            dx = wvln / 3
            dz = dx  # Use square pixels
            zz, xx = self.make_pixel_grid(x_axis, z_axis, xlims, zlims, dx, dz)
            c_x = np.round(((oc_x - x_axis.min())/(x_axis.max() - x_axis.min())) * (x_axis.shape[0]-1))
            c_z = np.round(((oc_z - z_axis.min())/(z_axis.max() - z_axis.min())) * (z_axis.shape[0]-1))
            c_r = oc_r / dx
            for i in range(len(c_x)):
                temp[int(c_z[i]), int(c_x[i])] = 255
            temp = cv2.remap(temp, xx.astype('float32'), zz.astype('float32'), cv2.INTER_CUBIC)
            temp_indices = np.argsort(temp.ravel())[::-1]
            temp_indices = np.unravel_index(temp_indices, temp.shape)
            coord_list = []
            for i in range(len(c_x)):
                coord_plot = [temp_indices[1][i], temp_indices[0][i], c_r[i]]
                coord_list.append(coord_plot)
            coord_info.append(coord_list)
        elif dataset_id.split('_')[0] == 'resolution':
            if dataset_id.split('_')[-1] == 'simu':
                f = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_iq.hdf5'), "r")["US"]["US_DATASET0000"]
                fp = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/simulation/resolution_distorsion/resolution_distorsion_simu_phantom.hdf5'), "r")["US"]["US_DATASET0000"]
                fs = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5'), "r")["US"]["US_DATASET0000"]
            elif dataset_id.split('_')[-1] == 'expe':
                f = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/experiments/resolution_distorsion/resolution_distorsion_expe_dataset_iq.hdf5'), "r")["US"]["US_DATASET0000"]
                fp = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/experiments/resolution_distorsion/resolution_distorsion_expe_phantom.hdf5'), "r")["US"]["US_DATASET0000"]
                fs = h5py.File(os.path.join(self.PICMUS_hdf5_path, 'database/experiments/resolution_distorsion/resolution_distorsion_expe_scan.hdf5'), "r")["US"]["US_DATASET0000"]
            ss = np.array(f["sound_speed"],dtype = "float32")
            sf = np.array(f["sampling_frequency"],dtype = "float32")
            mf = np.array(f["modulation_frequency"],dtype = "float32")
            ele_pos = np.array(f["probe_geometry"])
            x_axis = np.array(fs["x_axis"],dtype = "float32")
            z_axis = np.array(fs["z_axis"],dtype = "float32")
            xlims = (ele_pos[0,0] , ele_pos[0,-1])
            zlims = np.array([5e-3, 55e-3])
            wvln = ss / mf
            dx = wvln / 3
            dz = dx  # Use square pixels
            zz, xx = self.make_pixel_grid(x_axis, z_axis, xlims, zlims, dx, dz)
            x_plot_info = np.array(fp["scatterers_positions"],dtype = "float32")[0,:]
            x_plot_info = np.round(((x_plot_info - x_axis.min())/(x_axis.max() - x_axis.min())) * (x_axis.shape[0]-1))
            z_plot_info = np.array(fp["scatterers_positions"],dtype = "float32")[2,:]
            z_plot_info = np.round((( z_plot_info - z_axis.min())/(z_axis.max() - z_axis.min())) * (z_axis.shape[0]-1))
            for i in range(len(x_plot_info)):
                temp[int(z_plot_info[i]), int(x_plot_info[i])] = 255
            temp = cv2.remap(temp, xx.astype('float32'), zz.astype('float32'), cv2.INTER_CUBIC)
            temp_indices = np.argsort(temp.ravel())[::-1]
            temp_indices = np.unravel_index(temp_indices, temp.shape)
            coord_list = []
            for i in range(len(x_plot_info)):
                coord_plot = [temp_indices[1][i], temp_indices[0][i]]
                coord_list.append(coord_plot)
            coord_info.append(coord_list)
        return coord_info, dx*1e3
    
    #annotation plot_lateral
    def save_plot_image(self, coord_info):
        os.makedirs(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2]), exist_ok=True)
        fig, ax = plt.subplots()
        ax.imshow(self.gt_image, 'gray')
        ax.axis('off')
        for i, coord in enumerate(coord_info):
            coord[0]  = coord[0]
            coord[1]  = coord[1]
            ax.plot(coord[0], coord[1], 'r.')
            ax.text(coord[0], coord[1] - 5, str(i+1), color='r')
       
        plt.savefig(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.png'))
        plt.savefig(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.eps'))
        plt.clf()
        plt.close()

    def save_roi_hist(self, roi, bg, num, datatype, dataform):
        if dataform == 'envelope':
            x = np.linspace(0,1, 256)
            plt.hist(roi,bins=x,alpha=0.8, density=True, color='darkorange', label= 'roi')
            plt.hist(bg,bins=x,alpha=0.8, density=True, color='steelblue', label='bg')
            plt.legend()
            plt.savefig(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-2], self.image_path.split('/')[-1],'Circle{0:04d}'.format(num+1) + '_{}'.format(datatype) + '_{}'.format(dataform) + '.png'))
            plt.savefig(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-2], self.image_path.split('/')[-1],'Circle{0:04d}'.format(num+1) + '_{}'.format(datatype) + '_{}'.format(dataform) + '.eps'))
        if dataform == 'gray':
            x = np.linspace(0,255, 256)
            plt.hist(roi,bins=x,alpha=0.8, color='darkorange', label= 'roi')
            plt.hist(bg,bins=x,alpha=0.8, color='steelblue', label='bg')
            plt.legend()
            plt.savefig(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-2], self.image_path.split('/')[-1],'Circle{0:04d}'.format(num+1) + '_{}'.format(datatype) + '_{}'.format(dataform) + '.png'))
            plt.savefig(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-2], self.image_path.split('/')[-1],'Circle{0:04d}'.format(num+1) + '_{}'.format(datatype) + '_{}'.format(dataform) + '.eps'))
        plt.clf()
        plt.close()

    # def plot_circle(self, coord_info):
    #     os.makedirs(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2]), exist_ok=True)
    #     os.makedirs(os.path.join(self.save_path, 'Metric', self.image_path.split('/')[-2]), exist_ok=True)
    #     os.makedirs(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-2], self.image_path.split('/')[-1]), exist_ok=True)
    #     fig, ax = plt.subplots()
    #     ax.imshow(self.gt_image, 'gray')
    #     ax.axis('off')
    #     theta = np.linspace(0, 2*np.pi, 100)
    #     for i, coords in enumerate(coord_info[0]):
    #         x_coord, y_coord, radius = coords[0], coords[1], coords[2]
    #         mask_out_1 = np.zeros_like(self.gt_image)
    #         mask_out_2 = np.zeros_like(self.gt_image)
    #         cv2.circle(mask_out_1, center = (round(x_coord), round(y_coord)), radius = round(radius)+5, color=1, thickness=-1)
    #         cv2.circle(mask_out_2, center = (round(x_coord), round(y_coord)), radius = int(1.2*np.sqrt(pow(radius-5,2)+pow(radius+5,2))), color=1, thickness=-1)
    #         mask = cv2.bitwise_xor(mask_out_1, mask_out_2)
    #         mask = np.where(mask==1)
    #         ax.text(x_coord, y_coord - radius - 25, 'Circle{}'.format(i+1), color='r')
    #         ax.plot(x_coord + radius * np.cos(theta), y_coord + radius * np.sin(theta), 'r')
    #         ax.plot(x_coord + (radius + 5) * np.cos(theta), y_coord + (radius + self.resolution) * np.sin(theta), 'steelblue')
    #         ax.plot(x_coord + 1.2*np.sqrt(pow(radius-5,2)+pow(radius+5,2)) * np.cos(theta), y_coord + 1.2*np.sqrt(pow(radius-5,2)+pow(radius+5,2)) * np.sin(theta), 'steelblue')
    #         ax.scatter(x=mask[1], y=mask[0], c='steelblue', s=0.5, alpha=0.1)
    #         ax.plot(x_coord + (radius - 5) * np.cos(theta), y_coord + (radius - 5) * np.sin(theta), color='darkorange')
    #         ax.fill(x_coord + (radius - 5) * np.cos(theta), y_coord + (radius - 5) * np.sin(theta),facecolor='darkorange', alpha=0.3)
    #     plt.savefig(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.png'))
    #     plt.savefig(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.eps'))
    #     plt.clf()
    #     plt.close()

    #     for i, coords in enumerate(coord_info[0]):
    #         gt_metric = self.compute_circle(self.gt_image, self.gt_envelope, coord_info, 'gt')
    #         input_metric = self.compute_circle(self.input_image, self.input_envelope, coord_info, 'input')
    #         pred_metric = self.compute_circle(self.pred_image,self.pred_envelope, coord_info, 'pred')

    #     df = pd.DataFrame([], columns=['Datatype', 'BG'] + gt_metric['circle'])
    #     for metric, metric_name in zip(['snr', 'cr_image', 'cr_envelope', 'cnr_image', 'cnr_envelope', 'gcnr', 'cr_prop'], ['SNR', 'CR_IMG', 'CR_ENV', 'CNR_IMG', 'CNR_ENV', 'GCNR', 'CR_Prop']):
    #         df = df.append(pd.Series([metric_name] + [''] * (1 + len(gt_metric['circle'])), index=df.columns), ignore_index=True)
    #         if metric == 'snr':
    #             df = df.append(pd.Series(['Input'] + input_metric['bg_snr'] + input_metric[metric], index=df.columns), ignore_index=True)
    #             df = df.append(pd.Series(['GT'] + gt_metric['bg_snr'] + gt_metric[metric], index=df.columns), ignore_index=True)
    #             df = df.append(pd.Series(['Pred'] + pred_metric['bg_snr'] + pred_metric[metric], index=df.columns), ignore_index=True)
    #         else:
    #             df = df.append(pd.Series(['Input'] + [''] + input_metric[metric], index=df.columns), ignore_index=True)
    #             df = df.append(pd.Series(['GT'] + [''] + gt_metric[metric], index=df.columns), ignore_index=True)
    #             df = df.append(pd.Series(['Pred'] + [''] + pred_metric[metric], index=df.columns), ignore_index=True)
    #     df.to_csv(os.path.join(self.save_path, 'Metric', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.csv'))
    #     return input_metric, gt_metric, pred_metric
    
    def plot_circle(self, coord_info):
        # os.makedirs(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2]), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'Metric', self.image_path.split('/')[-2]), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'Hist', self.image_path.split('/')[-2], self.image_path.split('/')[-1]), exist_ok=True)
        # fig, ax = plt.subplots()
        # ax.imshow(self.gt_image, 'gray')
        # ax.axis('off')
        # theta = np.linspace(0, 2*np.pi, 100)
        # for i, coords in enumerate(coord_info[0]):
        #     x_coord, y_coord, radius = coords[0], coords[1], coords[2]
        #     mask_out_1 = np.zeros_like(self.gt_image)
        #     mask_out_2 = np.zeros_like(self.gt_image)
        #     cv2.circle(mask_out_1, center = (round(x_coord), round(y_coord)), radius = round(radius)+5, color=1, thickness=-1)
        #     cv2.circle(mask_out_2, center = (round(x_coord), round(y_coord)), radius = int(1.2*np.sqrt(pow(radius-5,2)+pow(radius+5,2))), color=1, thickness=-1)
        #     mask = cv2.bitwise_xor(mask_out_1, mask_out_2)
        #     mask = np.where(mask==1)
        #     ax.text(x_coord, y_coord - radius - 25, 'Circle{}'.format(i+1), color='r')
        #     ax.plot(x_coord + radius * np.cos(theta), y_coord + radius * np.sin(theta), 'r')
        #     ax.plot(x_coord + (radius + 5) * np.cos(theta), y_coord + (radius + 5) * np.sin(theta), 'steelblue')
        #     ax.plot(x_coord + 1.2*np.sqrt(pow(radius-5,2)+pow(radius+5,2)) * np.cos(theta), y_coord + 1.2*np.sqrt(pow(radius-5,2)+pow(radius+5,2)) * np.sin(theta), 'steelblue')
        #     ax.scatter(x=mask[1], y=mask[0], c='steelblue', s=0.5, alpha=0.1)
        #     ax.plot(x_coord + (radius - 5) * np.cos(theta), y_coord + (radius - 5) * np.sin(theta), color='darkorange')
        #     ax.fill(x_coord + (radius - 5) * np.cos(theta), y_coord + (radius - 5) * np.sin(theta),facecolor='darkorange', alpha=0.3)
        # plt.savefig(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.png'))
        # plt.savefig(os.path.join(self.save_path, 'Annotation', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.eps'))
        # plt.clf()
        # plt.close()

        for i, coords in enumerate(coord_info[0]):
            gt_metric = self.compute_circle(self.gt_image, self.gt_envelope, coord_info, 'gt')
            input_metric = self.compute_circle(self.input_image, self.input_envelope, coord_info, 'input')
            pred_metric = self.compute_circle(self.pred_image,self.pred_envelope, coord_info, 'pred')

        df = pd.DataFrame([], columns=['Datatype', 'BG'] + gt_metric['circle'])
        for metric, metric_name in zip(['snr', 'cr_image', 'cr_envelope', 'cnr_image', 'cnr_envelope', 'gcnr', 'cr_prop'], ['SNR', 'CR_IMG', 'CR_ENV', 'CNR_IMG', 'CNR_ENV', 'GCNR', 'CR_Prop']):
            df = df.append(pd.Series([metric_name] + [''] * (1 + len(gt_metric['circle'])), index=df.columns), ignore_index=True)
            if metric == 'snr':
                df = df.append(pd.Series(['Input'] + input_metric['bg_snr'] + input_metric[metric], index=df.columns), ignore_index=True)
                df = df.append(pd.Series(['GT'] + gt_metric['bg_snr'] + gt_metric[metric], index=df.columns), ignore_index=True)
                df = df.append(pd.Series(['Pred'] + pred_metric['bg_snr'] + pred_metric[metric], index=df.columns), ignore_index=True)
            else:
                df = df.append(pd.Series(['Input'] + [''] + input_metric[metric], index=df.columns), ignore_index=True)
                df = df.append(pd.Series(['GT'] + [''] + gt_metric[metric], index=df.columns), ignore_index=True)
                df = df.append(pd.Series(['Pred'] + [''] + pred_metric[metric], index=df.columns), ignore_index=True)
        df.to_csv(os.path.join(self.save_path, 'Metric', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.csv'))
        return input_metric, gt_metric, pred_metric

    def compute_circle(self, image, envelope, coord_info, datatype):
        bg_snr_mean_list, bg_snr_list, snr_list, cr_image_list, cr_envelope_list, cnr_image_list, cnr_envelope_list, gcnr_list, cr_prop_list, circle_list = [], [], [], [], [], [], [], [], [], []
        for i, coords in enumerate(coord_info[0]):
            bg_roi_image, roi_image = self.roi_circle(image, coords[0], coords[1], coords[2])
            bg_roi_image, roi_image = np.round(bg_roi_image * 255), np.round(roi_image * 255)
            bg_roi_envelope, roi_envelope = self.roi_circle(envelope, coords[0], coords[1], coords[2])
            bg_snr_list.append(float(signal_to_noise(bg_roi_image)))
            snr_list.append(float(signal_to_noise(roi_image)))
            cr_image_list.append(float(contrast_gray(roi_image, bg_roi_image)))
            cr_envelope_list.append(float(contrast_envelope(roi_envelope, bg_roi_envelope)))
            cnr_image_list.append(float(contrast_to_noise_gray(roi_image, bg_roi_image)))
            cnr_envelope_list.append(float(contrast_to_noise_envelope(roi_envelope, bg_roi_envelope)))
            gcnr_list.append(float(generalized_contrast_to_noise(roi_envelope, bg_roi_envelope)))
            cr_prop_list.append(float(cr_prop_fn(roi_envelope, roi_image, bg_roi_envelope, bg_roi_image)))
            circle_list.append('Circle{0:04d}'.format(i+1))
            self.save_roi_hist(roi_image, bg_roi_image, i, datatype, 'gray')
            self.save_roi_hist(roi_envelope, bg_roi_envelope, i, datatype, 'envelope')
        bg_snr_mean_list = [sum(bg_snr_list)/len(bg_snr_list)]
        metric = dict(bg_snr=bg_snr_mean_list, snr=snr_list, cr_image=cr_image_list, cr_envelope=cr_envelope_list, cnr_image=cnr_image_list, cnr_envelope=cnr_envelope_list, gcnr=gcnr_list, cr_prop=cr_prop_list, circle=circle_list)
        return metric

    def roi_circle(self, input, x, y, r):
        bg_mask_in = np.zeros_like(input)
        bg_mask_out = np.zeros_like(input)
        cv2.circle(bg_mask_in, center = (round(x),round(y)), radius = round(r)+5, color=1, thickness=-1)
        cv2.circle(bg_mask_out, center = (round(x),round(y)), radius = int(1.2*np.sqrt(pow(r-5,2)+pow(r+5,2))), color=1, thickness=-1)
        bg_mask = cv2.bitwise_xor(bg_mask_in,bg_mask_out)
        bg_roi = bg_mask * input.numpy()
        bg_roi_data = bg_roi.flatten()[np.nonzero(bg_roi.flatten())]
        roi_mask_in = np.zeros_like(input)
        roi_mask_out = np.zeros_like(input)
        cv2.circle(roi_mask_out, center = (round(x),round(y)), radius = round(r)-5, color=1, thickness=-1)
        roi_mask = cv2.bitwise_xor(roi_mask_in,roi_mask_out)
        roi = roi_mask * input.numpy()
        roi_data = roi.flatten()[np.nonzero(roi.flatten())]
        return bg_roi_data, roi_data
    
    def visualize_rf_spectrum(self):
        os.makedirs(os.path.join(self.save_path, 'RealRFSpectrum', self.image_path.split('/')[-2]), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'ImagRFSpectrum', self.image_path.split('/')[-2]), exist_ok=True)

        if self.datatype == 'rf':
            spectrum_type = ['RealRFSpectrum', 'ImagRFSpectrum']
        elif self.datatype == 'rf_real':
            spectrum_type = ['RealRFSpectrum']
        elif self.datatype == 'rf_imag':
            spectrum_type = ['ImagRFSpectrum']

        spectrum_list = []
        for datatype, rf in zip(['input', 'gt', 'pred'], [self.input, self.gt, self.pred]):
            fft = np.fft.fft(rf, axis=2)
            shift_fft = np.fft.fftshift(fft, axes=2)
            spectrum = 20 * np.log(np.abs(shift_fft))
            spectrum_list.append(spectrum)
            for i in range(spectrum.shape[0]):
                plt.figure()
                plt.imshow(spectrum[i])
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_path, spectrum_type[i], self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '_{}'.format(datatype) + '.png'))
                plt.clf()
                plt.close()

        for i in range(spectrum.shape[0]):
            fig = plt.figure(tight_layout=True)
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            ax1.imshow(spectrum_list[0][i, :, :]);ax1.axis('off');ax1.set_title('Compound Image')
            ax2.imshow(spectrum_list[1][i, :, :]);ax2.axis('off');ax2.set_title('Input Image')
            ax3.imshow(spectrum_list[2][i, :, :]);ax3.axis('off');ax3.set_title('Estimated Image')
            plt.savefig(os.path.join(self.save_path, spectrum_type[i], self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.png'))
            plt.clf()
            plt.close()

    def visualize_image_spectrum(self):
        os.makedirs(os.path.join(self.save_path, 'ImageSpectrum', self.image_path.split('/')[-2]), exist_ok=True)
        spectrum_list = []
        for datatype, image in zip(['input', 'gt', 'pred'], [self.input_image, self.gt_image, self.pred_image]):
            fft = np.fft.fft2(image)
            shift_fft = np.fft.fftshift(fft)
            spectrum = 20 * np.log(np.abs(shift_fft))
            spectrum_list.append(spectrum)
            plt.figure()
            plt.imshow(spectrum)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, 'ImageSpectrum', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '_{}'.format(datatype) + '.png'))
            plt.clf()
            plt.close()
        fig = plt.figure(tight_layout=True)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.imshow(spectrum_list[0][:, :]);ax1.axis('off');ax1.set_title('Compound Image')
        ax2.imshow(spectrum_list[1][:, :]);ax2.axis('off');ax2.set_title('Input Image')
        ax3.imshow(spectrum_list[2][:, :]);ax3.axis('off');ax3.set_title('Estimated Image')
        plt.savefig(os.path.join(self.save_path, 'ImageSpectrum', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.png'))
        plt.clf()
        plt.close()

    def plot_spreadfunc(self, eval_info):
        os.makedirs(os.path.join(self.save_path, 'SpreadFunc', self.image_path.split('/')[-2],'Lateral'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'SpreadFunc', self.image_path.split('/')[-2],'Axial'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'Metric', self.image_path.split('/')[-2]), exist_ok=True)
        line, thr, point_num, rf_list = 8, 50, 50, []
        rf_list.append(self.input_envelope)
        rf_list.append(self.gt_envelope)
        rf_list.append(self.pred_envelope)

        input_lateral_fwhm_list, gt_lateral_fwhm_list, pred_lateral_fwhm_list,input_axial_fwhm_list, gt_axial_fwhm_list, pred_axial_fwhm_list, point_list = [], [], [], [], [], [], []
        for i, coord in enumerate(eval_info):
        # lateral
            input_signal_lateral = rf_list[0][coord[1], max(0, coord[0] - line):coord[0] + line]
            gt_signal_lateral = rf_list[1][coord[1], max(0, coord[0] - line):coord[0] + line]
            pred_signal_lateral = rf_list[2][coord[1], max(0, coord[0] - line):coord[0] + line]
            plot_coord_lateral = coord[0]
            signal_shape_lateral = input_signal_lateral.shape[0]
            
            f = interp1d(np.linspace(0, signal_shape_lateral - 1, num=input_signal_lateral.shape[0]), input_signal_lateral)
            input_signal_lateral = torch.tensor(f(np.linspace(0, signal_shape_lateral - 1, num=point_num)))
            f = interp1d(np.linspace(0, signal_shape_lateral - 1, num=gt_signal_lateral.shape[0]), gt_signal_lateral)
            gt_signal_lateral = torch.tensor(f(np.linspace(0, signal_shape_lateral - 1, num=point_num)))
            f = interp1d(np.linspace(0, signal_shape_lateral - 1, num=pred_signal_lateral.shape[0]), pred_signal_lateral)
            pred_signal_lateral = torch.tensor(f(np.linspace(0, signal_shape_lateral - 1, num=point_num)))

            # FWHM
            input_lateral_fwhm_list.append(self.compute_FWHM(input_signal_lateral) * signal_shape_lateral / point_num)
            gt_lateral_fwhm_list.append(self.compute_FWHM(gt_signal_lateral) * signal_shape_lateral / point_num)
            pred_lateral_fwhm_list.append(self.compute_FWHM(pred_signal_lateral) * signal_shape_lateral / point_num)
            
            # dB transform
            input_signal_lateral = 20 * torch.log10(input_signal_lateral)
            input_signal_lateral[input_signal_lateral < -thr] = -thr
            gt_signal_lateral = 20 * torch.log10(gt_signal_lateral)
            gt_signal_lateral[gt_signal_lateral < -thr] = -thr
            pred_signal_lateral = 20 * torch.log10(pred_signal_lateral)
            pred_signal_lateral[pred_signal_lateral < -thr] = -thr
            
            # Visualize
            plt.figure()
            plt.plot(np.linspace(max(0, plot_coord_lateral - line), min(rf_list[0].shape[0], plot_coord_lateral + line), num=point_num), input_signal_lateral, marker='.', ls=':', color='b', label='Input')
            plt.plot(np.linspace(max(0, plot_coord_lateral - line), min(rf_list[0].shape[0], plot_coord_lateral + line), num=point_num), gt_signal_lateral, marker='.', ls=':', color='g', label='GT')
            plt.plot(np.linspace(max(0, plot_coord_lateral - line), min(rf_list[0].shape[0], plot_coord_lateral + line), num=point_num), pred_signal_lateral, marker='.', ls=':', color='r', label='Pred')
            plt.tight_layout()
            plt.ylim([-thr-5, 0])
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(self.save_path, 'SpreadFunc', self.image_path.split('/')[-2], 'Lateral', self.image_path.split('/')[-1] + '_{0:04}.png'.format(i+1)))
            plt.savefig(os.path.join(self.save_path, 'SpreadFunc', self.image_path.split('/')[-2], 'Lateral', self.image_path.split('/')[-1] + '_{0:04}.eps'.format(i+1)))
            plt.clf()
            plt.close()

        # axial
            input_signal_axial = rf_list[0][max(0, coord[1] - line):min(rf_list[0].shape[0], coord[1] + line), coord[0]]
            gt_signal_axial = rf_list[1][max(0, coord[1] - line):min(rf_list[1].shape[0], coord[1] + line), coord[0]]
            pred_signal_axial = rf_list[2][max(0, coord[1] - line):min(rf_list[2].shape[0], coord[1] + line), coord[0]]
            plot_coord_axial = coord[1]
            signal_shape_axial = input_signal_axial.shape[0]

            f = interp1d(np.linspace(0, signal_shape_axial - 1, num=input_signal_axial.shape[0]), input_signal_axial)
            input_signal_axial = torch.tensor(f(np.linspace(0, signal_shape_axial - 1, num=point_num)))
            f = interp1d(np.linspace(0, signal_shape_axial - 1, num=gt_signal_axial.shape[0]), gt_signal_axial)
            gt_signal_axial = torch.tensor(f(np.linspace(0, signal_shape_axial - 1, num=point_num)))
            f = interp1d(np.linspace(0, signal_shape_axial - 1, num=pred_signal_axial.shape[0]), pred_signal_axial)
            pred_signal_axial = torch.tensor(f(np.linspace(0, signal_shape_axial - 1, num=point_num)))

            # FWHM
            input_axial_fwhm_list.append(self.compute_FWHM(input_signal_axial) * signal_shape_axial / point_num)
            gt_axial_fwhm_list.append(self.compute_FWHM(gt_signal_axial) * signal_shape_axial / point_num)
            pred_axial_fwhm_list.append(self.compute_FWHM(pred_signal_axial) * signal_shape_axial / point_num)

            # dB transform
            input_signal_axial = 20 * torch.log10(input_signal_axial)
            input_signal_axial[input_signal_axial < -thr] = -thr
            gt_signal_axial = 20 * torch.log10(gt_signal_axial)
            gt_signal_axial[gt_signal_axial < -thr] = -thr
            pred_signal_axial = 20 * torch.log10(pred_signal_axial)
            pred_signal_axial[pred_signal_axial < -thr] = -thr

            # Visualize
            plt.figure()
            plt.plot(np.linspace(max(0, plot_coord_axial - line), min(rf_list[0].shape[0], plot_coord_axial + line), num=point_num), input_signal_axial, marker='.', ls=':', color='b', label='Input')
            plt.plot(np.linspace(max(0, plot_coord_axial - line), min(rf_list[0].shape[0], plot_coord_axial + line), num=point_num), gt_signal_axial, marker='.', ls=':', color='g', label='GT')
            plt.plot(np.linspace(max(0, plot_coord_axial - line), min(rf_list[0].shape[0], plot_coord_axial + line), num=point_num), pred_signal_axial, marker='.', ls=':', color='r', label='Pred')
            plt.tight_layout()
            plt.ylim([-thr-5, 0])
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(self.save_path, 'SpreadFunc', self.image_path.split('/')[-2], 'Axial', self.image_path.split('/')[-1] + '_{0:04}.png'.format(i+1)))
            plt.savefig(os.path.join(self.save_path, 'SpreadFunc', self.image_path.split('/')[-2], 'Axial', self.image_path.split('/')[-1] + '_{0:04}.eps'.format(i+1)))
            plt.clf()
            plt.close()
            
            point_list.append('Point{0:04d}'.format(i+1))
            
        df = pd.DataFrame([], columns=['Datatype'] + point_list)
        df = df.append(pd.Series(['Lateral'] + ['']*len(gt_lateral_fwhm_list), index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['GT'] + gt_lateral_fwhm_list, index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['Input'] + input_lateral_fwhm_list, index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['Pred'] + pred_lateral_fwhm_list, index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['Axial'] + ['']*len(gt_axial_fwhm_list), index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['GT'] + gt_axial_fwhm_list, index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['Input'] + input_axial_fwhm_list, index=df.columns), ignore_index=True)
        df = df.append(pd.Series(['Pred'] + pred_axial_fwhm_list, index=df.columns), ignore_index=True)
        df.to_csv(os.path.join(self.save_path, 'Metric', self.image_path.split('/')[-2], self.image_path.split('/')[-1] + '.csv'))
        
        return input_lateral_fwhm_list, gt_lateral_fwhm_list, pred_lateral_fwhm_list, input_axial_fwhm_list, gt_axial_fwhm_list, pred_axial_fwhm_list


    def compute_FWHM(self, signal):
        line = signal.shape[0] // 2
        mask = np.nonzero(signal >= (torch.max(signal) * 0.5))
        max_index = torch.where(signal == torch.max(signal))[0][0].item()
        mask_index = mask[mask == max_index].item()
        index = mask_index - 1
        for i in range(1, line):
            if mask[mask == index].shape[0] == 0 or index == 0:
                break
            index = index - 1
        fwhm_min_index = index
        index = mask_index + 1
        for i in range(1, line):
            if mask[mask == index].shape[0] == 0 or index == signal.shape[0] - 1:
                break                    
            index = index + 1
        fwhm_max_index = index
        if (fwhm_min_index == 0 and mask[0] == 0) or (fwhm_max_index == signal.shape[0] - 1 and mask[-1] == signal.shape[0] - 1):
            fwhm = 0
        else:
            fwhm = fwhm_max_index - fwhm_min_index
        return fwhm * self.resolution

    def summary_result(self):
        for dataset_id in ['resolution_distorsion_expe', 'resolution_distorsion_simu']:
            df = pd.DataFrame([], columns=['Datatype_Lateral', 'All', 'Num. of FWHM', 'Per. of FWHM', 'Used Num. of FWHM', 'Avg_FWHM'])
            input_lateral_fwhm =  np.concatenate(self.eval_data[dataset_id]['input_lateral_fwhm'])
            gt_lateral_fwhm = np.concatenate(self.eval_data[dataset_id]['gt_lateral_fwhm'])
            pred_lateral_fwhm = np.concatenate(self.eval_data[dataset_id]['pred_lateral_fwhm'])       
            idx_lateral = np.logical_and(input_lateral_fwhm, gt_lateral_fwhm)
            idx_lateral = np.logical_and(idx_lateral, pred_lateral_fwhm)
            for metric, datatype in zip(['gt_lateral_fwhm', 'input_lateral_fwhm', 'pred_lateral_fwhm'], ['GT', 'Input', 'Pred']):
                self.eval_data[dataset_id][metric] = np.array([e for inner_list in self.eval_data[dataset_id][metric] for e in inner_list])
                all_count = self.eval_data[dataset_id][metric].shape[0]
                fwhm_count = (self.eval_data[dataset_id][metric] != 0).sum()
                used_count = np.count_nonzero(idx_lateral)
                avg_fwhm = np.mean(self.eval_data[dataset_id][metric][idx_lateral])
                df = df.append(pd.Series([datatype, all_count, fwhm_count, fwhm_count / all_count * 100, used_count, avg_fwhm], index=df.columns), ignore_index=True)
            df.to_csv(os.path.join(self.save_path, dataset_id + '_Lateral.csv'))
            if dataset_id == 'resolution_distorsion_expe':
                lateral_result = np.array(df)
            else:
                lateral_result += np.array(df)

            df = pd.DataFrame([], columns=['Datatype_Axial', 'All', 'Num. of FWHM', 'Per. of FWHM', 'Used Num. of FWHM', 'Avg_FWHM'])
            input_axial_fwhm =  np.concatenate(self.eval_data[dataset_id]['input_axial_fwhm'])
            gt_axial_fwhm = np.concatenate(self.eval_data[dataset_id]['gt_axial_fwhm'])
            pred_axial_fwhm = np.concatenate(self.eval_data[dataset_id]['pred_axial_fwhm'])
            idx_axial = np.logical_and(input_axial_fwhm, gt_axial_fwhm)
            idx_axial = np.logical_and(idx_axial, pred_axial_fwhm)       
            for metric, datatype in zip(['gt_axial_fwhm', 'input_axial_fwhm', 'pred_axial_fwhm'], ['GT', 'Input', 'Pred']):
                self.eval_data[dataset_id][metric] = np.array([e for inner_list in self.eval_data[dataset_id][metric] for e in inner_list])
                all_count = self.eval_data[dataset_id][metric].shape[0]
                fwhm_count = (self.eval_data[dataset_id][metric] != 0).sum()
                used_count = np.count_nonzero(idx_axial)
                avg_fwhm = np.mean(self.eval_data[dataset_id][metric][idx_axial])
                df = df.append(pd.Series([datatype, all_count, fwhm_count, fwhm_count / all_count * 100, used_count, avg_fwhm], index=df.columns), ignore_index=True)
            df.to_csv(os.path.join(self.save_path, dataset_id + '_Axial.csv'))
            if dataset_id == 'resolution_distorsion_expe':
                axial_result = np.array(df)
            else:
                axial_result += np.array(df)

        lateral_result = np.delete(lateral_result, 0, axis=1)
        lateral_result = np.delete(lateral_result, 2, axis=1)
        lateral_result[:, -1] = lateral_result[:, -1]/2
        df = pd.DataFrame(lateral_result, columns=['All', 'Num. of FWHM', 'Used Num. of FWHM', 'Avg_FWHM'])
        df.insert(0, 'Datatype_Lateral', ['GT', 'Input', 'Pred'])
        df.to_csv(os.path.join(self.save_path, 'Result_PICMUS_Lateral.csv'))
        
        axial_result = np.delete(axial_result, 0, axis=1)
        axial_result = np.delete(axial_result, 2, axis=1)
        axial_result[:, -1] = axial_result[:, -1]/2
        df = pd.DataFrame(axial_result, columns=['All', 'Num. of FWHM', 'Used Num. of FWHM', 'Avg_FWHM'])
        df.insert(0, 'Datatype_Axial', ['GT', 'Input', 'Pred'])
        df.to_csv(os.path.join(self.save_path, 'Result_PICMUS_Axail.csv'))

        for dataset_id in ['contrast_speckle_expe', 'contrast_speckle_simu']:
            df = pd.DataFrame([], columns=['Datatype', 'BG_SNR', 'SNR', 'CR_IMG', 'CR_ENV', 'CNR_IMG', 'CNR_ENV', 'GCNR', 'CR_Prop'])
            for datatype_name, datatype in zip(['GT', 'Input', 'Pred'], ['gt', 'input', 'pred']):
                metric_list = []
                for metric, metric_name in zip(['bg_snr', 'snr', 'cr_image', 'cr_envelope', 'cnr_image', 'cnr_envelope', 'gcnr', 'cr_prop'], ['BG_SNR', 'SNR', 'CR_IMG', 'CR_ENV', 'CNR_IMG', 'CNR_ENV', 'GCNR', 'CR_Prop']):
                    self.eval_data[dataset_id][datatype + '_' + metric] = np.array([e for inner_list in self.eval_data[dataset_id][datatype + '_' + metric] for e in inner_list])
                    metric_list.append(np.mean(self.eval_data[dataset_id][datatype + '_' + metric]))
                df = df.append(pd.Series([datatype_name] + metric_list, index=df.columns), ignore_index=True)
            df.to_csv(os.path.join(self.save_path, dataset_id + '.csv'))
            if dataset_id == 'contrast_speckle_expe':
                contrast_result = np.array(df)
            else:
                contrast_result += np.array(df)
        contrast_result = np.delete(contrast_result, 0, axis=1)
        contrast_result = contrast_result / 2
        df = pd.DataFrame(contrast_result, columns=['BG_SNR', 'SNR', 'CR_IMG', 'CR_ENV', 'CNR_IMG', 'CNR_ENV', 'GCNR', 'CR_Prop'])
        df.insert(0, 'Datatype', ['GT', 'Input', 'Pred'])
        df.to_csv(os.path.join(self.save_path, 'Result_PICMUS_Contrast.csv'))