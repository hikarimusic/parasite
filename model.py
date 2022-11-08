import torch
from torch import nn
from torchsummary import summary


class Conv_Bn_Act(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == "mish":
            self.act = nn.Mish()
        elif activation == "leaky":
            self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Res(nn.Module):
    def __init__(self, channels, down1=False):
        super().__init__()
        if down1 == True:
            self.conv1 = Conv_Bn_Act(channels, channels//2, 1, 1, "mish")
            self.conv2 = Conv_Bn_Act(channels//2, channels, 3, 1, "mish")
        else:
            self.conv1 = Conv_Bn_Act(channels, channels, 1, 1, "mish")
            self.conv2 = Conv_Bn_Act(channels, channels, 3, 1, "mish")
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = x + x2
        return x3


class CSP(nn.Module):
    def __init__(self, channels, blocks, down1=False):
        super().__init__()
        if down1 == True:
            self.conv1 = Conv_Bn_Act(channels, 2*channels, 3, 2, "mish")
            self.conv2 = Conv_Bn_Act(2*channels, 2*channels, 1, 1, "mish")
            self.conv3 = Conv_Bn_Act(2*channels, 2*channels, 1, 1, "mish")
            self.resx = Res(2*channels, down1=True)
            self.conv4 = Conv_Bn_Act(2*channels, 2*channels, 1, 1, "mish")
            self.conv5 = Conv_Bn_Act(4*channels, 2*channels, 1, 1, "mish")
        else:
            self.conv1 = Conv_Bn_Act(channels, 2*channels, 3, 2, "mish")
            self.conv2 = Conv_Bn_Act(2*channels, channels, 1, 1, "mish")
            self.conv3 = Conv_Bn_Act(2*channels, channels, 1, 1, "mish")
            self.res_list = [Res(channels) for i in range(blocks)]
            self.resx = nn.Sequential(*self.res_list)
            self.conv4 = Conv_Bn_Act(channels, channels, 1, 1, "mish")
            self.conv5 = Conv_Bn_Act(2*channels, 2*channels, 1, 1, "mish")
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        x4 = self.resx(x3)
        x5 = self.conv4(x4)
        x6 = torch.cat((x2, x5), 1)
        x7 = self.conv5(x6)
        return x7


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = Conv_Bn_Act(3, 32, 3, 1, "mish")
        self.down1 = CSP(32, 1, down1=True)
        self.down2 = CSP(64, 2)
        self.down3 = CSP(128, 8)
        self.down4 = CSP(256, 8)
        self.down5 = CSP(512, 4)

    def forward(self, x):
        x1 = self.feat(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        return x4, x5, x6


class Neck_SPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Act(1024, 512, 1, 1, "leaky")
        self.conv2 = Conv_Bn_Act(512, 1024, 3, 1, "leaky")
        self.conv3 = Conv_Bn_Act(1024, 512, 1, 1, "leaky")
        self.maxpool1 = nn.MaxPool2d(5, 1, (5 - 1) // 2)
        self.maxpool2 = nn.MaxPool2d(9, 1, (9 - 1) // 2)
        self.maxpool3 = nn.MaxPool2d(13, 1, (13 - 1) // 2)
        self.conv4 = Conv_Bn_Act(2048, 512, 1, 1, "leaky")
        self.conv5 = Conv_Bn_Act(512, 1024, 3, 1, "leaky")
        self.conv6 = Conv_Bn_Act(1024, 512, 1, 1, "leaky")
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.maxpool1(x3)
        x5 = self.maxpool2(x3)
        x6 = self.maxpool3(x3)
        x7 = torch.cat((x3, x4, x5, x6), 1)
        x8 = self.conv4(x7)
        x9 = self.conv5(x8)
        x10 = self.conv6(x9)
        return x10


class Neck_FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = Conv_Bn_Act(512, 256, 1, 1, "leaky")
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv1_2 = Conv_Bn_Act(512, 256, 1, 1, "leaky")
        self.conv1_3 = Conv_Bn_Act(512, 256, 1, 1, "leaky")
        self.conv1_4 = Conv_Bn_Act(256, 512, 3, 1, "leaky")
        self.conv1_5 = Conv_Bn_Act(512, 256, 1, 1, "leaky")
        self.conv1_6 = Conv_Bn_Act(256, 512, 3, 1, "leaky")
        self.conv1_7 = Conv_Bn_Act(512, 256, 1, 1, "leaky")
        self.conv2_1 = Conv_Bn_Act(256, 128, 1, 1, "leaky")
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv2_2 = Conv_Bn_Act(256, 128, 1, 1, "leaky")
        self.conv2_3 = Conv_Bn_Act(256, 128, 1, 1, "leaky")
        self.conv2_4 = Conv_Bn_Act(128, 256, 3, 1, "leaky")
        self.conv2_5 = Conv_Bn_Act(256, 128, 1, 1, "leaky")
        self.conv2_6 = Conv_Bn_Act(128, 256, 3, 1, "leaky")
        self.conv2_7 = Conv_Bn_Act(256, 128, 1, 1, "leaky")
    
    def forward(self, xa, xb, xc):
        x1 = self.conv1_1(xc)
        x2 = self.upsample1(x1)
        x3 = self.conv1_2(xb)
        x4 = torch.cat((x3, x2), 1)
        x5 = self.conv1_3(x4)
        x6 = self.conv1_4(x5)
        x7 = self.conv1_5(x6)
        x8 = self.conv1_6(x7)
        x9 = self.conv1_7(x8)
        x10 = self.conv2_1(x9)
        x11 = self.upsample2(x10)
        x12 = self.conv2_2(xa)
        x13 = torch.cat((x12, x11), 1)
        x14 = self.conv2_3(x13)
        x15 = self.conv2_4(x14)
        x16 = self.conv2_5(x15)
        x17 = self.conv2_6(x16)
        x18 = self.conv2_7(x17)
        return x18, x9, xc


class Neck_PAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = Conv_Bn_Act(128, 256, 3, 2, "leaky")
        self.conv1_2 = Conv_Bn_Act(512, 256, 1, 1, "leaky")
        self.conv1_3 = Conv_Bn_Act(256, 512, 3, 1, "leaky")
        self.conv1_4 = Conv_Bn_Act(512, 256, 1, 1, "leaky")
        self.conv1_5 = Conv_Bn_Act(256, 512, 3, 1, "leaky")
        self.conv1_6 = Conv_Bn_Act(512, 256, 1, 1, "leaky")
        self.conv2_1 = Conv_Bn_Act(256, 512, 3, 2, "leaky")
        self.conv2_2 = Conv_Bn_Act(1024, 512, 1, 1, "leaky")
        self.conv2_3 = Conv_Bn_Act(512, 1024, 3, 1, "leaky")
        self.conv2_4 = Conv_Bn_Act(1024, 512, 1, 1, "leaky")
        self.conv2_5 = Conv_Bn_Act(512, 1024, 3, 1, "leaky")
        self.conv2_6 = Conv_Bn_Act(1024, 512, 1, 1, "leaky")

    def forward(self, xa, xb, xc):
        x1 = self.conv1_1(xa)
        x2 = torch.cat((x1, xb), 1)
        x3 = self.conv1_2(x2)
        x4 = self.conv1_3(x3)
        x5 = self.conv1_4(x4)
        x6 = self.conv1_5(x5)
        x7 = self.conv1_6(x6)
        x8 = self.conv2_1(x7)
        x9 = torch.cat((x8, xc), 1)
        x10 = self.conv2_2(x9)
        x11 = self.conv2_3(x10)
        x12 = self.conv2_4(x11)
        x13 = self.conv2_5(x12)
        x14 = self.conv2_6(x13)
        return xa, x7, x14


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.spp = Neck_SPP()
        self.fpn = Neck_FPN()
        self.pan = Neck_PAN()

    def forward(self, xa, xb, xc):
        xc = self.spp(xc)
        xa, xb, xc = self.fpn(xa, xb, xc)
        xa, xb, xc = self.pan(xa, xb, xc)
        return xa, xb, xc


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = Conv_Bn_Act(128, 256, 3, 1, "leaky")
        self.conv1_2 = nn.Conv2d(256, 255, 1, 1)
        self.conv2_1 = Conv_Bn_Act(256, 512, 3, 1, "leaky")
        self.conv2_2 = nn.Conv2d(512, 255, 1, 1)
        self.conv3_1 = Conv_Bn_Act(512, 1024, 3, 1, "leaky")
        self.conv3_2 = nn.Conv2d(1024, 255, 1, 1)

    def forward(self, xa, xb, xc):
        xa = self.conv1_1(xa)
        xa = self.conv1_2(xa)
        xb = self.conv2_1(xb)
        xb = self.conv2_2(xb)
        xc = self.conv3_1(xc)
        xc = self.conv3_2(xc)
        return xa, xb, xc


class Yolov4(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head()
    
    def forward(self, x):
        xa, xb, xc = self.backbone(x)
        xa, xb, xc = self.neck(xa, xb, xc)
        xa, xb, xc = self.head(xa, xb, xc)
        return xa, xb, xc


if __name__ == '__main__':
    model = Yolov4().to("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.rand(3, 608, 608).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    output = model(input)
    print(f"Input size:\n    {input.size()}")
    print(f"Output size:\n    {[output[0].size(), output[1].size(), output[2].size()]}")
    summary(model, (3, 608, 608))
