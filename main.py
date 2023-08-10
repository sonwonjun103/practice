import random
import os
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from utils.config import CFG
from utils.transform import *
from utils.seed import *
from data.dataset import CustomDataset
from model.direction_model import *
from model.three_direction_model import *

from training import training
from save_fig import save_fig
from test.inference import *

def main():
    # git 변경사항
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"device : {device}")

    # set seed
    seed_everything(CFG['SEED'])

    axial_train = pd.read_excel(f"D:\\새 폴더\\Entropy\\T1\\Axial\\T1_axial_entropy_info.xlsx")
    sagittal_train = pd.read_excel(f"D:\\새 폴더\\Entropy\\T1\\Sagittal\\T1_sagittal_entropy_info.xlsx")
    coronal_train = pd.read_excel(f"D:\\새 폴더\\Entropy\\T1\\Coronal\\T1_coronal_entropy_info.xlsx")

    axial_test = pd.read_excel(f"D:\\새 폴더\\Entropy\\T1\\Axial\\T1_axial_test_entropy_info.xlsx")
    sagittal_test = pd.read_excel(f"D:\\새 폴더\\Entropy\\T1\\Sagittal\\T1_sagittal_test_entropy_info.xlsx")
    coronal_test = pd.read_excel(f"D:\\새 폴더\\Entropy\\T1\\Coronal\\T1_coronal_test_entropy_info.xlsx")

    print(f"Complete Load File")
    
    print(f"Start Making DataSet")
    #make dataset
    train_dataset=CustomDataset(axial_train, sagittal_train, coronal_train, train_transform)
    train_dataloader=DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)

    print(f" Complete Train Dataset!")

    test_dataset = CustomDataset(axial_test, sagittal_test, coronal_test, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

    print(f" Complete Test Dataset\n")

    #Load model
    _amodel=axial_model().cuda()
    _smodel=sagittal_model().cuda()
    _cmodel=coronal_model().cuda()
    emodel=ensemble_model().cuda()

    amodel=nn.DataParallel(_amodel).to(device)
    smodel=nn.DataParallel(_smodel).to(device)
    cmodel=nn.DataParallel(_cmodel).to(device)
    #emodel=nn.DataParallel(_emodel).to(device)

    print(f"Complete Model")

    # Optimizer
    parameters=list(amodel.parameters())+list(smodel.parameters())+list(cmodel.parameters())+list(emodel.parameters())
    optimizer=torch.optim.Adam(parameters, lr=CFG['LR'])

    # LOSS
    asd = train_dataset._return_asd()
    tc = train_dataset._return_tc()

    tc_weight = asd / (asd+tc)
    asd_weight = tc / (asd+tc)
    weight = torch.Tensor([tc_weight, asd_weight]).to(device)
    print(f"Weight : {weight}")
    loss_fn=torch.nn.CrossEntropyLoss(weight).to(device)
    
    #Training & Test (Validation)
    # set test label
    test_label = []
    for p in coronal_test['PATH'].values:
        if 'ASD' in p:
            test_label.append(np.array([0,1]))
        else:
            test_label.append(np.array([1,0]))

    test_auc_label=[]

    for i in test_label:
        test_auc_label.append(i[1])
    
    print(f"Start Trainig and Test\n")
    #training
    train_loss, train_acc, test_loss, test_acc, senstivity, specificity, AUC, ACC=training(train_dataloader, test_dataloader, amodel, smodel, cmodel, emodel, optimizer, loss_fn, test_auc_label, device)

    #save information
    save_fig(train_loss, 1)
    save_fig(train_acc, 2)
    save_fig(test_loss, 3)
    save_fig(senstivity, 4)
    save_fig(specificity, 5)
    save_fig(AUC, 6)

    #last epoch inference
    inference_loop(test_dataloader, test_auc_label)
if __name__=='__main__':
    main()