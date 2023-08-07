from typing import Optional, Union, List
import torch.nn.functional as F
import torch.nn as nn
from .util import *

class Unet1D2D_featuremap(nn.Module):
    def __init__(self, in_channels, classes):
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        super(Unet1D2D_featuremap, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.encoder_2d = ResNet34_2DEncoder(
            in_channels = self.in_channels
        )
        self.decoder_2d = UnetDecoder2D(
            decoder_channels=decoder_channels[0]
        )
        self.encoder_1d = ResNet34_1DEncoder(
            in_channels = self.in_channels
        )
        self.decoder_1d = UnetDecoder1D(
            decoder_channels=decoder_channels[0]
        )

        self.conv =  nn.Sequential(
                    conv3x3(32, 16),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True)
        )
                    

        # self.outputer_1d = OutputChanelHead_1d(
        #     in_channels=16,
        #     out_channels=classes,
        #     activation=None,
        #     kernel_size=3,
        # )

        self.outputer_1d = OutputChanelHead(
            in_channels=16,
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )

        self.outputer_2d = OutputChanelHead(
            in_channels=16,
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )
        self.outputer_1d2d = OutputChanelHead(
            in_channels=16,
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        features = []
        x1 = self.encoder_1d(x)
        x1 = self.decoder_1d(x1) 
        bs = x1.shape[0]
        # output_1d = convert_2d_to_1d(x1)
        # output_1d = self.outputer_1d(output_1d)
        # output_1d = convert_1d_to_2d(output_1d, bs)
        # output_1d = nn.Tanh()(output_1d)
        output_1d = nn.Tanh()(self.outputer_1d(x1))
        features.append(output_1d)
        x2 = self.encoder_2d(x)
        x2 = self.decoder_2d(x2)
        output_2d = nn.Tanh()(self.outputer_2d(x2))
        features.append(output_2d)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        logits = self.outputer_1d2d(x)
        return logits, features

class Unet1D2D(nn.Module):
    def __init__(self, in_channels, classes):
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        super(Unet1D2D, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.encoder_2d = ResNet34_2DEncoder(
            in_channels = self.in_channels
        )
        self.decoder_2d = UnetDecoder2D(
            decoder_channels=decoder_channels[0]
        )
        self.encoder_1d = ResNet34_1DEncoder(
            in_channels = self.in_channels
        )
        self.decoder_1d = UnetDecoder1D(
            decoder_channels=decoder_channels[0]
        )

        self.conv =  nn.Sequential(
                    conv3x3(32, 16),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True)
        )
                    
        self.outputer = OutputChanelHead(
            in_channels=16,
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        x1 = self.encoder_1d(x)
        x1 = self.decoder_1d(x1)

        x2 = self.encoder_2d(x)
        x2 = self.decoder_2d(x2)

        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        logits = self.outputer(x)
        return logits


class Unet2D(nn.Module):
    def __init__(self, in_channels, classes):
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        super(Unet2D, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.encoder = ResNet34_2DEncoder(
            in_channels = self.in_channels
        )
        self.decoder = UnetDecoder2D(
            decoder_channels=decoder_channels[0]
        )
        self.outputer = OutputChanelHead(
            in_channels=16,
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        logits = self.outputer(x)
        return logits
    

class Unet1D(nn.Module):
    def __init__(self, in_channels, classes):
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        super(Unet1D, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.encoder = ResNet34_1DEncoder(in_channels = self.in_channels)
        self.decoder = UnetDecoder1D(
            decoder_channels=decoder_channels[0]
        )
        self.outputer = OutputChanelHead(
            in_channels=16,
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        logits = self.outputer(x)
        return logits


class ResNet34_2DEncoder(nn.Module):
    def __init__(self, in_channels):
        super(ResNet34_2DEncoder, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = 64
        self.conv1 = nn.Conv2d(
            self.in_channels, self.mid_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mid_channels = 64
        self.layer1 = self._make_layer(ResBlock2D, 64, 3, stride=1)
        self.layer2 = self._make_layer(ResBlock2D, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResBlock2D, 256, 6, stride=2)
        self.layer4 = self._make_layer(ResBlock2D, 512, 3, stride=2)
        # 重みを初期化する。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride):
        layers = []
        # 最初の Residual Block
        layers.append(block(self.mid_channels, channels, stride))
        # 残りの Residual Block
        self.mid_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.mid_channels, channels))
        return nn.Sequential(*layers)\

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features


class ResNet34_1DEncoder(nn.Module):
    def __init__(self, in_channels):
        super(ResNet34_1DEncoder, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = 64
        self.conv1 = nn.Conv1d(
            self.in_channels, self.mid_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.mid_channels = 64
        self.layer1 = self._make_layer(ResBlock1D, 64, 3, stride=1)
        self.layer2 = self._make_layer(ResBlock1D, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResBlock1D, 256, 6, stride=2)
        self.layer4 = self._make_layer(ResBlock1D, 512, 3, stride=2)
        # 重みを初期化する。
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride):
        layers = []
        # 最初の Residual Block
        layers.append(block(self.mid_channels, channels, stride))
        # 残りの Residual Block
        self.mid_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.mid_channels, channels))
        return nn.Sequential(*layers)\

    def forward(self, x):
        features = []
        bs = x.shape[0]
        x = convert_2d_to_1d(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = convert_1d_to_2d(x, bs)
        features.append(x)
        x = convert_2d_to_1d(x)
        x = self.maxpool(x)
        x = convert_1d_to_2d(x, bs)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features


class UnetDecoder2D(nn.Module):
    def __init__(
            self,
            decoder_channels,
    ):
        super().__init__()
        self.decorder_channels = decoder_channels
        self.block1 = DecoderBlock2D(512, decoder_channels[0], decoder_channels[0])
        self.block2 = DecoderBlock2D(decoder_channels[0], decoder_channels[1], decoder_channels[1])
        self.block3 = DecoderBlock2D(decoder_channels[1], decoder_channels[2], decoder_channels[2])
        self.block4 = DecoderBlock2D(decoder_channels[2], decoder_channels[2], decoder_channels[3])
        self.block5 = DecoderBlock2D(decoder_channels[3], 0, decoder_channels[4])
        self.blocks = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5
        )
        # 重みを初期化する。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        features = features[::-1]  # reverse channels to start from head of encoder
        skips = features[1:]
        x = features[0]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


class UnetDecoder1D(nn.Module):
    def __init__(
            self,
            decoder_channels,
    ):
        super().__init__()
        self.decorder_channels = decoder_channels
        self.block1 = DecoderBlock1D(512, decoder_channels[0], decoder_channels[0])
        self.block2 = DecoderBlock1D(decoder_channels[0], decoder_channels[1], decoder_channels[1])
        self.block3 = DecoderBlock1D(decoder_channels[1], decoder_channels[2], decoder_channels[2])
        self.block4 = DecoderBlock1D(decoder_channels[2], decoder_channels[2], decoder_channels[3])
        self.block5 = DecoderBlock1D(decoder_channels[3], 0, decoder_channels[4])
        self.blocks = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5
        )
        # 重みを初期化する。
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        features = features[::-1]  # reverse channels to start from head of encoder
        skips = features[1:]
        x = features[0]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x