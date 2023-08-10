import timm
import torch
import torch.nn as nn

from utils.config import CFG

class axial_model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model=timm.create_model('resnet101', pretrained=pretrained)
        self.model.fc=nn.Identity()

    def forward(self, axial):
        x=self.model(axial)
        #print(x.shape)
        if x.shape[0]==CFG['BATCH_SIZE']*20//2:
            x=torch.reshape(x, (CFG['BATCH_SIZE']//2, 20, x.shape[1]))
        else:
            x=torch.reshape(x, (CFG['BATCH_SIZE']//4, 20, x.shape[1]))

        return x
    
class sagittal_model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model=timm.create_model('resnet101', pretrained=pretrained)
        self.model.fc=nn.Identity()

    def forward(self, sagittal):
        x=self.model(sagittal)
        if x.shape[0]==CFG['BATCH_SIZE']*20//2:
            x=torch.reshape(x, (CFG['BATCH_SIZE']//2, 20, x.shape[1]))
        else:
            x=torch.reshape(x, (CFG['BATCH_SIZE']//4, 20, x.shape[1]))
        return x
    
class coronal_model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model=timm.create_model('resnet101', pretrained=pretrained)
        self.model.fc=nn.Identity()

    def forward(self, coronal):
        x=self.model(coronal)

        if x.shape[0]==CFG['BATCH_SIZE']*20//2:
            x=torch.reshape(x, (CFG['BATCH_SIZE']//2, 20, x.shape[1]))
        else:
            x=torch.reshape(x, (CFG['BATCH_SIZE']//4, 20, x.shape[1]))
        return x