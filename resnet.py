import torch
import torch.nn as nn

#-----------------------------------------convolutional block
class convBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.c(x))
#-------------------------------------------------------------

#-----------------------------------Bottleneck block (residual)
class residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        res_channels = in_channels // 4
        stride = 1


        self.projection = in_channels != out_channels
        if self.projection:
            self.p = convBlock(in_channels, out_channels, 1, 2, 0)
            stride = 2
            res_channels = in_channels // 2

        if first:
            self.p = convBlock(in_channels, out_channels, 1, 1, 0)
            stride = 1
            res_channels = in_channels

        self.c1 = convBlock(in_channels, res_channels, 1, 1, 0)
        self.c2 = convBlock(res_channels, res_channels, 3, stride, 1)
        self.c3 = convBlock(res_channels, out_channels, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self,x):
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)

        if self.projection:
            x = self.p(x)

        h = self.relu(torch.add(f, x))
        return h
#-------------------------------------------------------------
#-------------------------------------------------------ResNet
class resNet(nn.Module):
    def __init__(self, no_blocks, in_channels = 3, classes = 1000):
        super().__init__()
        out_features = [256, 512, 1024, 2048]
        self.blocks = nn.ModuleList([residualBlock(64, 256, True)])

        for i in range(len(out_features)):
            if i > 0:
                self.blocks.append(residualBlock(out_features[i-1], out_features[i]))
            for _ in range(no_blocks[i]-1):
                self.blocks.append(residualBlock(out_features[i], out_features[i]))
        
        self.conv1 = convBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
# Modellpéldány létrehozása
model = resNet([3, 4, 6, 3])
input_data = torch.randn(1, 3, 224, 224)
output = model(input_data)
print(output.shape)
