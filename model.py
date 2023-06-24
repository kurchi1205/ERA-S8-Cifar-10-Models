import torch.nn as nn
import torch.nn.functional as F
import torch


class NetBN(nn.Module):
    def __init__(self) -> None:
        super(NetBN, self).__init__()
        
        self.base_channel = 8
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.base_channel, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.base_channel),
            nn.Dropout(0.02)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel, out_channels=self.base_channel, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.base_channel),
            nn.Dropout(0.02)
        )

        self.trans_block1 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel, out_channels=self.base_channel * 2, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.base_channel * 2)
        )

        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 2, out_channels=self.base_channel * 2, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.base_channel * 2),
            nn.Dropout(0.02)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 2, out_channels=self.base_channel * 2, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.base_channel * 2),
            nn.Dropout(0.02)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 2, out_channels=self.base_channel * 2, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.base_channel * 2),
            nn.Dropout(0.02)
        )

        self.trans_block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 2, out_channels=self.base_channel * 4, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.base_channel * 4)
        )
        self.pool2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 4, out_channels=self.base_channel * 4, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.base_channel * 4),
            nn.Dropout(0.02)
        )

        self.conv_block7 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 4, out_channels=self.base_channel * 4, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.base_channel * 4),
            nn.Dropout(0.02)
        )

        self.conv_block8 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 4, out_channels=self.base_channel * 4, kernel_size=(3,3), padding=1, bias=False),
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        )

        self.out = nn.Conv2d(in_channels=self.base_channel * 4, out_channels=10, kernel_size=(1,1), bias=False)

    def forward(self, x):
        x = self.conv_block1(x) # rin = 1 rout = 3
        x = x + self.conv_block2(x) # rin = 3 rout = 5
        x = self.trans_block1(x) # rin = 5 rout = 5
        x = self.pool1(x) # rin = 5 rout = 6
        x = x + self.conv_block3(x) # rin = 6 rout = 10
        x = x + self.conv_block4(x) # rin = 10 rout = 14
        x = x + self.conv_block5(x) # rin = 14 rout = 18
        x = self.trans_block2(x) # rin = 18 rout = 18
        x = self.pool2(x) # rin = 18 rout = 20
        x = x + self.conv_block6(x) # rin = 20 rout = 28
        x = x + self.conv_block7(x) # rin = 28 rout = 36
        x = x + self.conv_block8(x) # rin = 36 rout = 44
        x = self.gap(x) 
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)




class NetLN(nn.Module):
    def __init__(self) -> None:
        super(NetLN, self).__init__()
        
        self.base_channel = 8
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.base_channel, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=1, num_channels=self.base_channel),
            nn.Dropout(0.02)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel, out_channels=self.base_channel, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=1, num_channels=self.base_channel),
            nn.Dropout(0.02)
        )

        self.trans_block1 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel, out_channels=self.base_channel * 2, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.base_channel * 2) 
        )

        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 2, out_channels=self.base_channel * 2, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=1, num_channels=self.base_channel * 2),
            nn.Dropout(0.02)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 2, out_channels=self.base_channel * 2, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=2, num_channels=self.base_channel * 2),
            nn.Dropout(0.02)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 2, out_channels=self.base_channel * 2, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=1, num_channels=self.base_channel * 2),
            nn.Dropout(0.02)
        )

        self.trans_block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 2, out_channels=self.base_channel * 4, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.base_channel * 4)
        )
        self.pool2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 4, out_channels=self.base_channel * 4, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=1, num_channels=self.base_channel * 4),
            nn.Dropout(0.02)
        )

        self.conv_block7 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 4, out_channels=self.base_channel * 4, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=1, num_channels=self.base_channel * 4),
            nn.Dropout(0.02)
        )

        self.conv_block8 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 4, out_channels=self.base_channel * 4, kernel_size=(3,3), padding=1, bias=False),
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        )

        self.out = nn.Conv2d(in_channels=self.base_channel * 4, out_channels=10, kernel_size=(1,1), bias=False)

    def forward(self, x):
        x = self.conv_block1(x) # rin = 1 rout = 3
        x = x + self.conv_block2(x) # rin = 3 rout = 5
        x = self.trans_block1(x) # rin = 5 rout = 5
        x = self.pool1(x) # rin = 5 rout = 6
        x = self.conv_block3(x) # rin = 6 rout = 10
        y = self.conv_block4(x) # rin = 10 rout = 14
        x = x + self.conv_block5(y) # rin = 14 rout = 18
        x = self.trans_block2(x) # rin = 18 rout = 18
        x = self.pool2(x) # rin = 18 rout = 20
        x = self.conv_block6(x) # rin = 20 rout = 28
        y = self.conv_block7(x) # rin = 28 rout = 36
        x = x + self.conv_block8(y) # rin = 36 rout = 44
        x = self.gap(x) 
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)    
    

    
    
    
class NetGN(nn.Module):
    def __init__(self) -> None:
        super(NetGN, self).__init__()
        
        self.base_channel = 8
        self.num_groups = 4
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.base_channel, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=self.num_groups, num_channels=self.base_channel),
            nn.Dropout(0.02)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel, out_channels=self.base_channel, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=self.num_groups, num_channels=self.base_channel),
            nn.Dropout(0.02)
        )

        self.trans_block1 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel, out_channels=self.base_channel * 2, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(num_groups=self.num_groups, num_channels=self.base_channel * 2) 
        )

        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 2, out_channels=self.base_channel * 2, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=self.num_groups, num_channels=self.base_channel * 2),
            nn.Dropout(0.02)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 2, out_channels=self.base_channel * 2, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=self.num_groups, num_channels=self.base_channel * 2),
            nn.Dropout(0.02)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 2, out_channels=self.base_channel * 2, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=self.num_groups, num_channels=self.base_channel * 2),
            nn.Dropout(0.02)
        )

        self.trans_block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 2, out_channels=self.base_channel * 4, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(num_groups=self.num_groups, num_channels=self.base_channel * 4)
        )
        self.pool2 = nn.MaxPool2d((2, 2), stride=2)

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 4, out_channels=self.base_channel * 4, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=self.num_groups, num_channels=self.base_channel * 4),
            nn.Dropout(0.02)
        )

        self.conv_block7 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 4, out_channels=self.base_channel * 4, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(num_groups=self.num_groups, num_channels=self.base_channel * 4),
            nn.Dropout(0.02)
        )

        self.conv_block8 = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channel * 4, out_channels=self.base_channel * 4, kernel_size=(3,3), padding=1, bias=False),
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        )

        self.out = nn.Conv2d(in_channels=self.base_channel * 4, out_channels=10, kernel_size=(1,1), bias=False)

    def forward(self, x):
        x = self.conv_block1(x) # rin = 1 rout = 3
        x = x + self.conv_block2(x) # rin = 3 rout = 5
        x = self.trans_block1(x) # rin = 5 rout = 5
        x = self.pool1(x) # rin = 5 rout = 6
        x = self.conv_block3(x) # rin = 6 rout = 10
        y = self.conv_block4(x) # rin = 10 rout = 14
        x = x + self.conv_block5(y) # rin = 14 rout = 18
        x = self.trans_block2(x) # rin = 18 rout = 18
        x = self.pool2(x) # rin = 18 rout = 20
        x = self.conv_block6(x) # rin = 20 rout = 28
        y = self.conv_block7(x) # rin = 28 rout = 36
        x = x + self.conv_block8(y) # rin = 36 rout = 44
        x = self.gap(x) 
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)