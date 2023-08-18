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
import json
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.functional import peak_signal_noise_ratio

from cnn_lib import *
from criterion.criterion import *
from utils import *
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from skimage.metrics import *
import optuna

class Tuner(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        # Epoch
        self.epochs = cfgs['train']['epochs']
        # CUDA
        self.device = cfgs['device']
        # Directory
        self.savedir = os.path.join(cfgs['root_path'], 'model', cfgs['target'], cfgs['exp_name'])
        self.checkpoint_path = cfgs['checkpoint_path']

        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.l2_loss = nn.MSELoss()
        self.rf_loss = RFLoss(cfgs)
        self.fourier_loss = FourierLoss(cfgs)
        
         
        self.is_loss_image = cfgs['train']['is_loss_image']
        self.is_loss_iq = cfgs['train']['is_loss_iq']
        self.is_loss_rf = cfgs['train']['is_loss_rf']
        self.is_loss_fourier = cfgs['train']['is_loss_fourier']
        self.is_loss_lpips = cfgs['train']['is_loss_lpips']
        self.is_loss_featuremap = cfgs['train']['is_loss_featuremap']
        self.alpha_amp = cfgs['train']['alpha_amp']
        self.alpha_pahse = cfgs['train']['alpha_phase']
        if cfgs['train']['tuning']:
            self.lpisp_loss = LPIPSLoss(cfgs)
            
        self.min_loss = float('inf')
        
        if self.checkpoint_path:
            checkpoint = load_checkpoint(self.checkpoint_path, self.device)
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.eval_info = json.load(open('coord_NEW_shidara.json', 'r'))
        self.save_path = os.path.join(cfgs['root_path'], 'result', cfgs['target'], cfgs['exp_name'])
        self.visualize_list = ['RealRFSpectrum', 'ImagRFSpectrum', 'ImageSpectrum', 'Image', 'train', 'val', 'SpreadFunc', 'Annotation', 'Metric', 'Hist']
        for visualize_name in self.visualize_list:
            visualize_save_path = os.path.join(self.save_path, visualize_name)
            os.makedirs(visualize_save_path, exist_ok=True)

        # Set tensorboard
        self.writer = SummaryWriter(os.path.join(cfgs['root_path'], 'log', cfgs['target'], cfgs['exp_name']))

        self.metric_list = ['loss', 'pred_ssim', 'pred_psnr', 'pred_lpips', 'input_ssim', 'input_psnr', 'input_lpips']

    def main(self, trainloader, valloader):
        trial_size = 1000   
        trial_epochs = 25  
        
        study = optuna.create_study()  
        tuner = self.Tuning(trainloader, valloader, trial_epochs)
        study.optimize(tuner, n_trials=trial_size, callbacks=[self.save])
        print(f"Best Object Value : {study.best_value}")
        print(f"Best Parameters : {study.best_params}")
                                             
    def Tuning(self, trainloader, valloader, traial_epochs):
        def objective(trial):
            print('Start Tuning...')
            self.net = select_model(self.cfgs).to(self.device)
            self.optimizer = select_optimizer(self.net, self.cfgs)
            alpha_amp = trial.suggest_float('alpha_amp', 0, 0)
            alpha_phase = trial.suggest_float('alpha_phase', 0, 0)
            beta_Envelope = trial.suggest_float('beta_Envelope', 0, 0)
            beta_Fourier = trial.suggest_float('beta_Fourier', 0, 0)

            meter_list = [AverageMeter(meter) for meter in self.metric_list]
            progress = ProgressMeter(traial_epochs, meter_list, prefix='Training: ')
            n_earlystop = 0
            for epoch in range(1, traial_epochs+1):
                # Inference
                results = self.trainer('train', trainloader, epoch, alpha_amp, alpha_phase, beta_Envelope, beta_Fourier)
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
            print('End Tuning')
            self.optimizer.zero_grad()
            self.net = None
            self.optimizer.step()
            return loss
        return objective

    def val(self, epoch, valloader):
        print('Start Validation...')
        meter_list = [AverageMeter(meter) for meter in self.metric_list]
        progress = ProgressMeter(1, meter_list, prefix='Validation: ')
        # Inference
        with torch.no_grad():
            results = self.trainer('val', valloader, epoch, 0, 0, 0, 0)
            for metric, meter in zip(self.metric_list, meter_list):
                meter.update(results[metric] / results['iter'])
            progress.display(1)
        return meter_list[0].avg / results['iter']

    def test(self, datatype, testloader):
        print('Start Evaluation...')
        self.error_df = make_df(self.metric_list)
        meter_list = [AverageMeter(meter) for meter in self.metric_list]
        progress = ProgressMeter(1, meter_list, prefix='Evaluation: ')
        # Inference
        with torch.no_grad():
            results = self.trainer(datatype, testloader, 1)
            for metric, meter in zip(self.metric_list, meter_list):
                meter.update(results[metric] / results['iter'])
            progress.display(1)
        self.error_df = write_df(self.error_df, results['loss'], results['pred_ssim'] / results['iter'], results['pred_psnr'] / results['iter'], results['pred_lpips'] / results['iter'], results['input_ssim'] / results['iter'], results['input_psnr'] / results['iter'], results['input_lpips'] / results['iter'])
        save_df(os.path.join(self.cfgs['root_path'], 'result', self.cfgs['target'], self.cfgs['exp_name'], datatype + '_error.csv'), self.error_df)
        print('End Evaluation...')

    def trainer(self, phase, dataloader, epoch, alpha_amp, alpha_phase, beta_Envelope, beta_Fourier):
        if phase == 'train':
            self.net.train()
        else:
            self.net.eval()
        print('Start Iteration...')
        results = dict(iter=0, loss=0, pred_ssim=0, pred_psnr=0, pred_lpips=0,input_ssim=0, input_psnr=0, input_lpips=0)
        loss_list = dict(loss=0, loss_iq=0, loss_rf=0, rf_envelope_loss=0, rf_phase_loss=0, loss_fourier=0, fourier_amp_loss=0, fourier_phase_loss=0, iteration=0, lpips_loss = 0)
        image_paths = []

        for data in tqdm(dataloader):
            loss_list['iteration'] += 1
            input = data['input'].to(self.device)
            gt = data['gt'].to(self.device)
            raw_data = data['raw_gt'].to(self.device)
            batch_size = gt.shape[0]
            results['iter'] += 1
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
            if phase == 'train':
                if self.is_loss_iq:
                    loss_iq = self.l1_loss(pred, gt)
                    loss_list['loss_iq'] += loss_iq.item()
                    loss += loss_iq

                if self.is_loss_rf:
                    loss_rf, rf_envelope_loss, rf_phase_loss = self.rf_loss(pred, gt)
                    loss_list['loss_rf'] += loss_rf.item()
                    loss_list['rf_envelope_loss'] += rf_envelope_loss.item()
                    loss_list['rf_phase_loss'] += rf_phase_loss.item()
                    loss += beta_Envelope * alpha_amp * rf_envelope_loss

                if self.is_loss_fourier:
                    loss_fourier, fourier_amp_loss, fourier_phase_loss = self.fourier_loss(pred, gt)
                    loss_list['loss_fourier'] += loss_fourier.item()
                    loss_list['fourier_amp_loss'] += fourier_amp_loss.item()
                    loss_list['fourier_phase_loss'] += fourier_phase_loss.item()
                    loss += (alpha_amp *  beta_Fourier * fourier_amp_loss) + (alpha_phase * beta_Fourier * fourier_phase_loss)

                if self.is_loss_lpips:
                    lpips_loss = self.lpisp_loss(pred, gt)
                    loss_list['lpips_loss'] += lpips_loss.item()
                    loss += lpips_loss
                
                if self.is_loss_image:
                    image_loss = self.l1_loss(transform2image(pred, raw_data, 'rf'), transform2image(gt, raw_data, 'rf'))
                    loss += image_loss

                if self.is_loss_featuremap:
                    for i in range(len(featuremap)):
                        loss_feature = self.l1_loss(transforms.CenterCrop((raw_data.shape[2], raw_data.shape[3]))(featuremap[i]), gt)
                        loss += loss_feature
            else:
                lpips_loss = self.lpisp_loss(pred, gt)
                loss += lpips_loss
                
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_list['loss'] += loss.item()
            results['loss'] += loss.item()
        return results
    
    def save(self, study, frozen_trial):
        tuning_log = study.trials_dataframe()
        tuning_log.to_csv(os.path.join(self.save_path, 'tuning_log.csv'), sep='\t', index=None)