
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )

def conv2d_1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


def conv3x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1d_1x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(
        in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
    )


def convert_2d_to_1d(x_2d):
    bs = x_2d.shape[0]
    x_1d = x_2d.permute(0, 2, 1, 3)
    x_1d = torch.reshape(
        x_1d, (bs*x_1d.shape[1], x_1d.shape[2], x_1d.shape[3]))
    return x_1d


def convert_1d_to_2d(x_1d, bs):
    x_1d = torch.reshape(
        x_1d, (bs, x_1d.shape[0]//bs, x_1d.shape[1], x_1d.shape[2])
        )
    x_2d = x_1d.permute(0, 2, 1, 3)
    return x_2d
    
class ResBlock2D(nn.Module):
    expansion = 1  # 出力のチャンネル数を入力のチャンネル数の何倍に拡大するか

    def __init__(
        self,
        in_channels,
        channels,
        stride=1
    ):
        super().__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

        # 入力と出力のチャンネル数が異なる場合、x をダウンサンプリングする。
        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv2d_1x1(in_channels, channels * self.expansion, stride),
                nn.BatchNorm2d(channels * self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out
    
class ResBlock1D(nn.Module):
    expansion = 1  # 出力のチャンネル数を入力のチャンネル数の何倍に拡大するか

    def __init__(
        self,
        in_channels,
        channels,
        stride=1
    ):
        super().__init__()
        self.conv1 = conv3x1(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)

        # 入力と出力のチャンネル数が異なる場合、x をダウンサンプリングする。
        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv1d_1x1(in_channels, channels * self.expansion, stride),
                nn.BatchNorm1d(channels * self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        bs = x.shape[0]
        x = convert_2d_to_1d(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        out = convert_1d_to_2d(out, bs)
        return out


class DecoderBlock2D(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            use_resblock =False,
    ):
        super().__init__()
        if use_batchnorm:
            self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )
        self.attention1 =  nn.Identity()
        
        if use_resblock:
            if use_batchnorm:
                self.conv2 = ResBlock2D()
            else:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                )
        else:
            if use_batchnorm:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                )

        self.attention2 =  nn.Identity()

    def forward(self, x, skip = None):
        x = self.conv1(x)
        if skip is not None:
            skip = F.interpolate(skip, size=(x.shape[2], x.shape[3]), mode="bilinear")
            x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x

class DecoderBlock1D(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            use_resblock =False,
    ):
        super().__init__()
        if use_batchnorm:
            self.conv1 = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )
        
        if use_resblock:
            if use_batchnorm:
                self.conv2 = ResBlock1D()
            else:
                self.conv2 = nn.Sequential(
                    nn.Conv1d(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                )
        else:
            if use_batchnorm:
                self.conv2 = nn.Sequential(
                    nn.Conv1d(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                self.conv2 = nn.Sequential(
                    nn.Conv1d(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                )
    def forward(self, x, skip = None):
        bs = x.shape[0]
        x = convert_2d_to_1d(x)
        x = self.conv1(x)
        x = convert_1d_to_2d(x, bs)
        if skip is not None:
            skip = F.interpolate(skip, size=(x.shape[2], x.shape[3]), mode="bilinear")
            x = torch.cat([x, skip], dim=1)
        x = convert_2d_to_1d(x)
        x = self.conv2(x)
        x = convert_1d_to_2d(x, bs)
        return x
    
class OutputChanelHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        activation = Activation(activation)
        super().__init__(conv2d, activation)
        
class OutputChanelHead_1d(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None):
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        activation = Activation(activation)
        super().__init__(conv1d, activation)



class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)
