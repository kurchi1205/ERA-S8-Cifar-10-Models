import torch.nn as nn
import torch.nn.functional as F
import torch

## Session 6
class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      # Extracting 16 features using 3x3 kernel but keeping size same
      self.conv1 = nn.Conv2d(1, 16, 3, padding=1) #rin = 1 rout= 3
      # Performing batchNormalization
      self.bn1 = nn.BatchNorm2d(16)
      # Performing maxPooling assuming 1st level of features are extracted
      self.pool1 = nn.MaxPool2d(2, 2); #rin = 3 rout= 4
      # Avoiding overfitting
      self.dropout1 = nn.Dropout(0.10);
      # Extracting 2nd level of features
      self.conv2 = nn.Conv2d(16, 32, 3, padding=1) #rin = 4 rout= 8
      # Performing batchNormalization
      self.bn2 = nn.BatchNorm2d(32)
      # Performing maxPooling assuming 2nd level of features are extracted
      self.pool2 = nn.MaxPool2d(2, 2); #rin = 8 rout= 10
      # Avoiding overfitting
      self.dropout2 = nn.Dropout(0.10);
      # Performing fully connected but maintaining spatial information
      self.conv3 = nn.Conv2d(32, 64, 1) #rin = 10 rout= 10
      self.bn3 = nn.BatchNorm2d(64)
      # Extract the important information and increase receptive field
      self.pool3 = nn.MaxPool2d(2, 2); #rin = 10 rout = 14
      # Getting info for 10 classes
      self.conv4 = nn.Conv2d(64, 10, 3) #rin = 14 rout= 30
      
  def forward(self, x):
    x = self.pool1(self.bn1(F.relu(self.conv1(x))))
    x = self.dropout1(x)
    x = self.pool2(self.bn2(F.relu(self.conv2(x))))
    x = self.dropout2(x)
    x = self.pool3(self.bn3(F.relu(self.conv3(x))))
    x = self.conv4(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=1)



## Session 7
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False), # rin=1 rout=3
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False), # rin=3 rout=5
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),  # rin=5 rout=7
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) 

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11  # rin=7 rout=8
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # rin=8 rout=8
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),  # rin=8 rout=12
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),  # rin=12 rout=16
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) 

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # rin=16 rout=16
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) 

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    
    
    
    
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False), # rin = 1 rout = 3
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),  # rin = 3 rout = 5
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),  # rin = 5 rout = 7
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # rin = 7 rout = 8
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # rin = 8 rout = 8
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),  # rin = 8 rout = 12
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=30, kernel_size=(3, 3), padding=0, bias=False),  # rin = 12 rout = 16
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=20, kernel_size=(1, 1), padding=0, bias=False),  # rin = 16 rout = 16
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1
        self.out = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)  # rin = 16 rout = 16

        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    

class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        # Input Block 28  >>> 64
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False), # rin = 1 rout = 3
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.02)
        ) # output_size = 26 >>> 62

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=1, bias=False), # rin = 3 rout = 5
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.02)
        )

        # TRANSITION BLOCK 1
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), # rin = 6 rout = 6
            nn.BatchNorm2d(10),
            nn.ReLU(),
        ) # output_size = 11 >>> 29
        self.pool1 = nn.MaxPool2d(2, 2)# rin = 5 rout = 6
        

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False), # rin = 6 rout = 10
            nn.BatchNorm2d(20),
            nn.Dropout(0.02)
        )

        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), # rin = 12 rout = 12
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2) # rin = 10 rout = 12
        

        # CONVOLUTION BLOCK 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False), # rin = 12 rout = 20
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.02)
        )
        self.out = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3, 3), padding=0, bias=False) # rin = 20 rout = 28
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        ) # output_size = 1


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
       
        x = self.pool1(x)
        x = self.convblock4(x)
        
        x = self.convblock5(x)
        
        x = self.pool2(x)
        x = self.convblock6(x)
        
        x = self.convblock7(x)
        x = self.out(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)





## Session 8
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