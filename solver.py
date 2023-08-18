import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import random
import math
import json
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image
import pandas as pd

from cnn_lib import *
from criterion.criterion import *
from utils import *
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from skimage.metrics import *

class Solver(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        # Epoch
        self.epochs = cfgs['train']['epochs']
        # CUDA
        self.device = cfgs['device']
        # Directory
        self.savedir = os.path.join(cfgs['root_path'], 'model', cfgs['target'], cfgs['exp_name'])
        self.checkpoint_path = cfgs['checkpoint_path']

        self.net = select_model(cfgs).to(self.device)
        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.l2_loss = nn.MSELoss()
        self.rf_loss = RFLoss(cfgs)
        self.fourier_loss = FourierLoss(cfgs)
        
        self.is_loss_image = cfgs['train']['is_loss_image']
        self.is_loss_iq = cfgs['train']['is_loss_iq']
        self.is_loss_rf = cfgs['train']['is_loss_rf']
        self.is_loss_fourier = cfgs['train']['is_loss_fourier']
        self.is_loss_featuremap = cfgs['train']['is_loss_featuremap']
            
        self.min_loss = float('inf')
        self.optimizer = select_optimizer(self.net, cfgs)

        if self.checkpoint_path:
            checkpoint = load_checkpoint(self.checkpoint_path, self.device)
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.eval_info = json.load(open('coord.json', 'r'))
        self.save_path = os.path.join(cfgs['root_path'], 'result', cfgs['target'], cfgs['exp_name'])
        self.visualize_list = ['RealRFSpectrum', 'ImagRFSpectrum', 'ImageSpectrum', 'Image', 'train', 'val', 'SpreadFunc', 'Annotation', 'Metric', 'Hist']
        for visualize_name in self.visualize_list:
            visualize_save_path = os.path.join(self.save_path, visualize_name)
            os.makedirs(visualize_save_path, exist_ok=True)

        # Set tensorboard
        self.writer = SummaryWriter(os.path.join(cfgs['root_path'], 'log', cfgs['target'], cfgs['exp_name']))

        self.metric_list = ['loss', 'pred_ssim', 'pred_psnr', 'pred_lpips', 'input_ssim', 'input_psnr', 'input_lpips']

    def train(self, trainloader, valloader):
        print('Start Training...')
        meter_list = [AverageMeter(meter) for meter in self.metric_list]
        progress = ProgressMeter(self.cfgs['train']['epochs'], meter_list, prefix='Training: ')
        n_earlystop = 0
        for epoch in range(1, self.epochs + 1):
            # Inference
            results = self.trainer('train', trainloader, epoch)
            for metric, meter in zip(self.metric_list, meter_list):
                meter.update(results[metric] / results['iter'])
            progress.display(epoch)
            for meter in meter_list:
                meter.reset()
            # Validation
            loss = self.val(epoch, valloader)
            # Save Checkpoint
            if self.min_loss > loss:
                n_earlystop = 0
                self.min_loss = loss
                save_checkpoint(self.cfgs, self.net, self.optimizer, epoch)
            else:
                n_earlystop += 1
            if n_earlystop == 20:
                break
        print('End Training')

    def val(self, epoch, valloader):
        print('Start Validation...')
        meter_list = [AverageMeter(meter) for meter in self.metric_list]
        progress = ProgressMeter(1, meter_list, prefix='Validation: ')
        # Inference
        with torch.no_grad():
            results = self.trainer('val', valloader, epoch)
            for metric, meter in zip(self.metric_list, meter_list):
                meter.update(results[metric] / results['iter'])
            progress.display(1)
        return meter_list[0].avg / results['iter']

    def PICMUS(self, epoch, dataloader):
        print('Start PICMUS...')
        # Inference
        with torch.no_grad():
            results = self.trainer('PICMUS', dataloader, epoch)

    def test(self, phase, phantom, testloader):
        print('Start Evaluation...')
        self.error_df = make_df(self.metric_list)
        meter_list = [AverageMeter(meter) for meter in self.metric_list]
        progress = ProgressMeter(1, meter_list, prefix='Evaluation: ')
        # Inference
        with torch.no_grad():
            results = self.trainer(phase, testloader, 1, phantom)
            for metric, meter in zip(self.metric_list, meter_list):
                meter.update(results[metric] / results['iter'])
            progress.display(1)
        self.error_df = write_df(self.error_df, results['loss'], results['pred_ssim'] / results['iter'], results['pred_psnr'] / results['iter'], results['pred_lpips'] / results['iter'], results['input_ssim'] / results['iter'], results['input_psnr'] / results['iter'], results['input_lpips'] / results['iter'])
        save_df(os.path.join(self.cfgs['root_path'], 'result', self.cfgs['target'], self.cfgs['exp_name'], phantom + '_error.csv'), self.error_df)
        print('End Evaluation...')

    def trainer(self, phase, dataloader, epoch, phantom = 'all'):
        if phase == 'train':
            self.net.train()
        else:
            self.net.eval()
        print('Start Iteration...')
        results = dict(iter=0, loss=0, pred_ssim=0, pred_psnr=0, pred_lpips=0,input_ssim=0, input_psnr=0, input_lpips=0)
        df = pd.DataFrame([], columns=['ImageName', 'SSIM_pred', 'PSNR_pred', 'LPIPS_pred','SSIM_input', 'PSNR_input', 'LPIPS_input'])
        evaluater = Evaluate(self.cfgs, self.save_path, self.eval_info)
        PICMUS_evaluater = PICMUS_Evaluate(self.cfgs, self.save_path)
        loss_list = dict(loss=0, loss_iq=0, loss_rf=0, rf_envelope_loss=0, rf_phase_loss=0, loss_fourier=0, fourier_amp_loss=0, fourier_phase_loss=0, iteration=0, lpips_loss = 0)
        image_paths = []

        for data in tqdm(dataloader):
            loss_list['iteration'] += 1
            input = data['input'].to(self.device)
            gt = data['gt'].to(self.device)
            raw_data = data['raw_gt'].to(self.device)
            batch_size = gt.shape[0]
            results['iter'] += 1

            if self.cfgs['model'] == 'Li':
                input = torch.squeeze(input)
                gt = torch.squeeze(gt)
                pred = self.net(torch.unsqueeze(input[:,0,:,:], 1))
                if phase == 'PICMUS':
                    input = transforms.functional.crop(input, 14, 13, raw_data.shape[2], raw_data.shape[3])
                    pred = transforms.functional.crop(pred, 14, 13, raw_data.shape[2], raw_data.shape[3])
                    gt = transforms.functional.crop(gt, 14, 13, raw_data.shape[2], raw_data.shape[3])
                else:    
                    input = transforms.CenterCrop((raw_data.shape[2], raw_data.shape[3]))(input)
                    pred = transforms.CenterCrop((raw_data.shape[2], raw_data.shape[3]))(pred)
                    gt = transforms.CenterCrop((raw_data.shape[2], raw_data.shape[3]))(gt)
                loss = 0
                image_loss = self.l1_loss(transform2image(torch.squeeze(pred), raw_data, 'rf_real'), transform2image(gt[:, 0, :, :], raw_data, 'rf_real'))
                loss += image_loss

                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if phase == 'test' or  phase == 'PICMUS':
                    ssim_func = StructuralSimilarityIndexMeasure()
                    psnr_func = PeakSignalNoiseRatio()
                    lpips_func = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize = True).cuda(self.cfgs['device'])
                    pred_lpips = lpips_func(stack_image(transform2image(gt[:,0,:,:], raw_data, 'rf_real'), transform2image(torch.squeeze(pred), raw_data, 'rf_real'))).item()
                    input_lpips = lpips_func(stack_image(transform2image(gt[:,0,:,:], raw_data, 'rf_real'), transform2image(input[:,0,:,:], raw_data, 'rf_real'))).item()

                input, gt, pred, raw_data = input.detach().cpu(), gt.detach().cpu(), pred.detach().cpu(), raw_data.detach().cpu()
              
                if phase == 'test':
                    pred_ssim, pred_psnr = ssim_func(transform2image(gt[:,0,:,:], raw_data, 'rf_real'), transform2image(torch.squeeze(pred), raw_data, 'rf_real')).item(), psnr_func(transform2image(gt[:,0,:,:], raw_data, 'rf_real'), transform2image(torch.squeeze(pred), raw_data, 'rf_real')).item()
                    input_ssim, input_psnr = ssim_func(transform2image(gt[:,0,:,:], raw_data, 'rf_real'), transform2image(input[:,0,:,:], raw_data, 'rf_real')).item(), psnr_func(transform2image(gt[:,0,:,:], raw_data, 'rf_real'), transform2image(input[:,0,:,:], raw_data, 'rf_real')).item()
                    for batch_idx in range(batch_size):
                        image_paths.append(data['input_path'])
                        evaluater.main(data['input_path'][batch_idx], input[batch_idx], gt[batch_idx], pred[batch_idx], raw_data[batch_idx], phantom)
                elif phase == 'PICMUS':
                    pred_ssim, pred_psnr = ssim_func(transform2image(gt[:,0,:,:], raw_data, 'rf_real'), transform2image(pred, raw_data, 'rf_real')).item(), psnr_func(transform2image(gt, raw_data, 'rf'), transform2image(pred, raw_data, 'rf')).item()
                    input_ssim, input_psnr = ssim_func(transform2image(gt[:,0,:,:], raw_data, 'rf_real'), transform2image(input[:,0,:,:], raw_data, 'rf_real')).item(), psnr_func(transform2image(gt, raw_data, 'rf'), transform2image(input, raw_data, 'rf')).item()
                    for batch_idx in range(batch_size):
                        image_paths.append(data['input_path'])
                        PICMUS_evaluater.main(data['input_path'][batch_idx], input[batch_idx], gt[batch_idx], pred[batch_idx], raw_data[batch_idx])
            else:
                if self.cfgs['channel'] == 2:
                    input = torch.squeeze(input)
                    gt = torch.squeeze(gt)
                if self.cfgs['datatype'] == 'envelope':
                    pred = nn.Sigmoid()(self.net(input))
                elif self.cfgs['datatype'] in ['us', 'rf', 'rf_real', 'rf_imag']:
                    if self.is_loss_featuremap:
                        pred,featuremap = self.net(input)
                        pred = nn.Tanh()(pred)
                    else:
                        pred = nn.Tanh()(self.net(input))

                if phase == 'PICMUS':
                    input = transforms.functional.crop(input, 14, 13, raw_data.shape[2], raw_data.shape[3])
                    pred = transforms.functional.crop(pred, 14, 13, raw_data.shape[2], raw_data.shape[3])
                    gt = transforms.functional.crop(gt, 14, 13, raw_data.shape[2], raw_data.shape[3])
                else:    
                    input = transforms.CenterCrop((raw_data.shape[2], raw_data.shape[3]))(input)
                    pred = transforms.CenterCrop((raw_data.shape[2], raw_data.shape[3]))(pred)
                    gt = transforms.CenterCrop((raw_data.shape[2], raw_data.shape[3]))(gt)
                loss = 0
                if self.is_loss_iq:
                    loss_iq = self.l1_loss(pred, gt)
                    loss_list['loss_iq'] += loss_iq.item()
                    loss += loss_iq

                if self.is_loss_rf:
                    loss_rf, rf_envelope_loss, rf_phase_loss = self.rf_loss(pred, gt)
                    loss_list['loss_rf'] += loss_rf.item()
                    loss_list['rf_envelope_loss'] += rf_envelope_loss.item()
                    loss_list['rf_phase_loss'] += rf_phase_loss.item()
                    loss += loss_rf

                if self.is_loss_fourier:
                    loss_fourier, fourier_amp_loss, fourier_phase_loss = self.fourier_loss(pred, gt)
                    loss_list['loss_fourier'] += loss_fourier.item()
                    loss_list['fourier_amp_loss'] += fourier_amp_loss.item()
                    loss_list['fourier_phase_loss'] += fourier_phase_loss.item()
                    loss += loss_fourier

                if self.is_loss_image:
                    image_loss = self.l1_loss(transform2image(pred, raw_data, 'rf'), transform2image(gt, raw_data, 'rf'))
                    loss += image_loss
                    
                if self.is_loss_featuremap:
                    for i in range(len(featuremap)):
                        loss_iq = self.l1_loss(transforms.CenterCrop((raw_data.shape[2], raw_data.shape[3]))(featuremap[i]), gt)
                        loss += loss_iq

                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                loss_list['loss'] += loss.item()

                plot = len(dataloader) * (epoch - 1) + loss_list['iteration']
                self.writer.add_scalar(phase + '/' + 'loss', loss.item(), plot)
                if self.is_loss_iq:
                    self.writer.add_scalar(phase + '/' + 'loss_iq', loss_iq.item(), plot)

                if self.is_loss_rf:
                    self.writer.add_scalar(phase + '/' + 'loss_rf', loss_rf.item(), plot)
                    self.writer.add_scalar(phase + '/' + 'rf_envelope_loss', rf_envelope_loss.item(), plot)
                    self.writer.add_scalar(phase + '/' + 'rf_phase_loss', rf_phase_loss.item(), plot)

                if self.is_loss_fourier:
                    self.writer.add_scalar(phase + '/' + 'loss_fourier', loss_fourier.item(), plot)
                    self.writer.add_scalar(phase + '/' + 'fourier_amp_loss', fourier_amp_loss.item(), plot)
                    self.writer.add_scalar(phase + '/' + 'fourier_phase_loss', fourier_phase_loss.item(), plot)

                if phase == 'test' or  phase == 'PICMUS':
                    ssim_func = StructuralSimilarityIndexMeasure()
                    psnr_func = PeakSignalNoiseRatio()
                    lpips_func = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize = True).cuda(self.cfgs['device'])
                    pred_lpips = lpips_func(stack_image(transform2image(gt, raw_data, 'rf')), stack_image(transform2image(pred, raw_data, 'rf'))).item()
                    input_lpips = lpips_func(stack_image(transform2image(gt, raw_data, 'rf')), stack_image(transform2image(input, raw_data, 'rf'))).item()

                input, gt, pred, raw_data = input.detach().cpu(), gt.detach().cpu(), pred.detach().cpu(), raw_data.detach().cpu()
                        
                if phase == 'test':
                    pred_ssim, pred_psnr = ssim_func(transform2image(gt, raw_data, 'rf'), transform2image(pred, raw_data, 'rf')).item(), psnr_func(transform2image(gt, raw_data, 'rf'), transform2image(pred, raw_data, 'rf')).item()
                    input_ssim, input_psnr = ssim_func(transform2image(gt, raw_data, 'rf'), transform2image(input, raw_data, 'rf')).item(), psnr_func(transform2image(gt, raw_data, 'rf'), transform2image(input, raw_data, 'rf')).item()
                    for batch_idx in range(batch_size):
                        image_paths.append(data['input_path'])
                        evaluater.main(data['input_path'][batch_idx], input[batch_idx], gt[batch_idx], pred[batch_idx], raw_data[batch_idx], phantom)
                elif phase == 'PICMUS':
                    pred_ssim, pred_psnr = ssim_func(transform2image(gt, raw_data, 'rf'), transform2image(pred, raw_data, 'rf')).item(), psnr_func(transform2image(gt, raw_data, 'rf'), transform2image(pred, raw_data, 'rf')).item()
                    input_ssim, input_psnr = ssim_func(transform2image(gt, raw_data, 'rf'), transform2image(input, raw_data, 'rf')).item(), psnr_func(transform2image(gt, raw_data, 'rf'), transform2image(input, raw_data, 'rf')).item()
                    for batch_idx in range(batch_size):
                        image_paths.append(data['input_path'])
                        PICMUS_evaluater.main(data['input_path'][batch_idx], input[batch_idx], gt[batch_idx], pred[batch_idx], raw_data[batch_idx])
                        
            results['loss'] += loss.item()
            
            if phase == 'test' or phase == 'PICMUS':
                results['pred_ssim'] += pred_ssim
                results['pred_psnr'] += pred_psnr
                results['pred_lpips'] += pred_lpips
                results['input_ssim'] += input_ssim
                results['input_psnr'] += input_psnr
                results['input_lpips'] += input_lpips


        self.writer.add_scalar(phase + '/' + 'epoch_loss', loss_list['loss'] / len(dataloader), epoch)
        self.writer.add_scalar(phase + '/' + 'epoch_loss_iq', loss_list['loss_iq'] / len(dataloader), epoch)
        self.writer.add_scalar(phase + '/' + 'epoch_loss_rf', loss_list['loss_rf'] / len(dataloader), epoch)
        self.writer.add_scalar(phase + '/' + 'epoch_rf_envelope_loss', loss_list['rf_envelope_loss'] / len(dataloader), epoch)
        self.writer.add_scalar(phase + '/' + 'epoch_rf_phase_loss', loss_list['rf_phase_loss'] / len(dataloader), epoch)
        self.writer.add_scalar(phase + '/' + 'epoch_loss_fourier', loss_list['loss_fourier'] / len(dataloader), epoch)
        self.writer.add_scalar(phase + '/' + 'epoch_fourier_amp_loss', loss_list['fourier_amp_loss'] / len(dataloader), epoch)
        self.writer.add_scalar(phase + '/' + 'epoch_fourier_phase_loss', loss_list['fourier_phase_loss'] / len(dataloader), epoch)
        self.writer.add_scalar(phase + '/' + 'epoch_fourier_phase_loss', loss_list['lpips_loss'] / len(dataloader), epoch)

        if phase == 'test' and phantom == 'qap':
            evaluater.summary_result()
        elif phase == 'PICMUS':
            PICMUS_evaluater.summary_result()
            
        if phase != 'test' and phase != 'PICMUS':
            idx = random.choice(list(range(0, input.shape[0])))
            if self.cfgs['model'] == 'Li':
                save_image(pixel_remap(transform2image(input[:, 0, :, :], raw_data, 'rf_real')[idx][0]), pixel_remap(transform2image(gt[:, 0, :, :], raw_data, 'rf_real')[idx][0]), pixel_remap(transform2image(torch.squeeze(pred), raw_data, 'rf_real')[idx][0]), os.path.join(self.save_path, '{}'.format(phase), 'epoch{0:4d}_'.format(epoch) + data['input_path'][idx].split('/')[-2] + '_' + data['input_path'][idx].split('/')[-1] + '.png'))
            else:
                save_image(pixel_remap(transform2image(input, raw_data, 'rf')[idx][0]), pixel_remap(transform2image(gt, raw_data, 'rf')[idx][0]), pixel_remap(transform2image(pred, raw_data, 'rf')[idx][0]), os.path.join(self.save_path, '{}'.format(phase), 'epoch{0:4d}_'.format(epoch) + data['input_path'][idx].split('/')[-2] + '_' + data['input_path'][idx].split('/')[-1] + '.png'))

        return results