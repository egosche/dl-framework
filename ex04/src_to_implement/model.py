import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.training = True

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),  # 3 = input channels  64 = output channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64, stride=1),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 512, stride=2),
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)  # may be wrong
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, input):
        layer0_output = self.layer0(input)
        layer1_output = self.layer1(layer0_output)
        gap_output = self.gap(layer1_output)
        fl_output = self.flatten(gap_output)
        fc_output = self.fc(fl_output)
        output = self.sigmoid(fc_output)

        return output


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.training = True

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        first_conv_output = self.conv1(input)
        first_bn_output = self.bn1(first_conv_output)
        first_relu_output = nn.ReLU()(first_bn_output)
        sec_conv_output = self.conv2(first_relu_output)
        sec_bn_output = self.bn2(sec_conv_output)
        sec_bn_output += shortcut  # Skip connection is added to the output of batch norm 2
        sec_relu_output = nn.ReLU()(sec_bn_output)

        return sec_relu_output
