import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import lpips

class FourierLoss(nn.Module):
    def __init__(self, cfgs):
        super(FourierLoss, self).__init__()
        self.criterion = nn.L1Loss() 
        self.alpha_amp = cfgs['train']['alpha_amp']
        self.alpha_pahse = cfgs['train']['alpha_phase']
        self.beta = cfgs['train']['beta_Fourier']
  
        # self.filter = torch.hann_window(window_length=384).to(cfgs['device'])
        # self.filter = torch.reshape(self.filter, [1, self.filter.shape[0], 1])

    def forward(self, input, target):
        input_fft_amp, input_fft_phase = self.fourier_transform_rf(input)
        target_fft_amp, target_fft_phase = self.fourier_transform_rf(target)
        amp_loss = self.criterion(torch.mean(target_fft_amp, dim = 2), torch.mean(input_fft_amp, dim = 2))
        phase_loss = self.criterion(target_fft_phase, input_fft_phase)
        return (self.alpha_amp *  self.beta * amp_loss) + (self.alpha_pahse *  self.beta * phase_loss), amp_loss, phase_loss
        
    def fourier_transform_rf(self, input):
        real= input[:, 0, :, :]
        fft_rf = torch.fft.fft(real, dim=1)
        amp = torch.abs(fft_rf)
        # phase = torch.angle(fft_rf)
        phase = torch.cos(torch.angle(fft_rf))
        # amp = torchvision.transforms.CenterCrop((500, amp.shape[2]))(amp)
        # phase = torchvision.transforms.CenterCrop((500, phase.shape[2]))(phase)
        return amp, phase

class RFLoss(nn.Module):
    def __init__(self, cfgs):
        super(RFLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.alpha_amp = cfgs['train']['alpha_amp']
        self.alpha_pahse = cfgs['train']['alpha_phase']
        self.beta = cfgs['train']['beta_Envelope']

    def forward(self, input, target):
        input_envelope, input_phase = self.rf(input)
        target_envelope, target_phase = self.rf(target)
        envelope_loss = self.criterion(target_envelope, input_envelope)
        phase_loss = self.criterion(target_phase, input_phase)
        return self.beta * self.alpha_amp * envelope_loss, envelope_loss, phase_loss
      
    def rf(self, input):
        real, imag = input[:, 0, :, :], input[:, 1, :, :]
        rf = torch.complex(real, imag)
        envelope = torch.abs(rf)
        phase = torch.cos(torch.angle(rf))
        return envelope, phase
    
class LPIPSLoss(nn.Module):
    def __init__(self, cfgs):
        super(LPIPSLoss, self).__init__()
        if cfgs['train']['lpips_net'] == 'vgg':
            self.criterion = lpips.LPIPS(net='vgg').cuda(cfgs['device'])
        else:
            self.criterion = lpips.LPIPS(net='alex').cuda(cfgs['device'])

    def forward(self, input, target):
        input = F.interpolate(input,scale_factor=(1,1), mode='bilinear')
        target = F.interpolate(target,scale_factor=(1,1), mode='bilinear')
        input= self.stack_image(input)
        target = self.stack_image(target)
        lpips_loss = self.criterion(input, target)
        return torch.mean(lpips_loss)
    
    def stack_image(self, input):
        real, imag = input[:, 0, :, :], input[:, 1, :, :]
        rf = torch.complex(real, imag)
        envelope = torch.abs(rf) / torch.max(torch.abs(rf))
        envelope = 20 * torch.log10(envelope)
        envelope[envelope < -60] = -60
        envelope = envelope + torch.abs(envelope.min())
        envelope = envelope / 60
        envelope = (envelope - 0.5) / 0.5
        envelope_list = []
        envelope_list.append(envelope)
        envelope_list.append(envelope)
        envelope_list.append(envelope)
        envelope = torch.stack(envelope_list, dim = 1)
        return envelope