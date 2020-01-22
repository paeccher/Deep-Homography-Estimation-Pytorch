import torch.nn as nn
import torch.nn.functional as F

class DeepHomographyModel(nn.Module):
    def __init__(self):
        super(DeepHomographyModel, self).__init__()
        self.conv_layer1 = nn.Conv2d(2,64,3,padding=1)
        self.conv_layer2 = nn.Conv2d(64,64,3,padding=1)
        self.conv_layer3 = nn.Conv2d(64,64,3,padding=1)
        self.conv_layer4 = nn.Conv2d(64,64,3,padding=1)
        self.conv_layer5 = nn.Conv2d(64,128,3,padding=1)        
        self.conv_layer6 = nn.Conv2d(128,128,3,padding=1)
        self.conv_layer7 = nn.Conv2d(128,128,3,padding=1)
        self.conv_layer8 = nn.Conv2d(128,128,3,padding=1)
        self.fc_layer1 = nn.Linear(128*16*16,1024)
        self.fc_layer2 = nn.Linear(1024,8)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.batch_norm6 = nn.BatchNorm2d(128)
        self.batch_norm7 = nn.BatchNorm2d(128)
        self.batch_norm8 = nn.BatchNorm2d(128)

    
    def forward(self,x):
        out = self.conv_layer1(x)
        out = self.batch_norm1(out) 
        out = F.relu(out)

        out = self.conv_layer2(out)
        out = self.batch_norm2(out) 
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv_layer3(out)
        out = self.batch_norm3(out) 
        out = F.relu(out)

        out = self.conv_layer4(out)
        out = self.batch_norm4(out) 
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv_layer5(out)
        out = self.batch_norm5(out) 
        out = F.relu(out)

        out = self.conv_layer6(out)
        out = self.batch_norm6(out) 
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.conv_layer7(out)
        out = self.batch_norm7(out) 
        out = F.relu(out)

        out = self.conv_layer8(out)
        out = self.batch_norm8(out) 
        out = F.relu(out)
        out = out.view(-1,128*16*16)

        out = self.fc_layer1(out)
        out = self.fc_layer2(out)
        return out
