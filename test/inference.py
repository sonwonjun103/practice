from save_fig import roc_auc
from utils.config import CFG
from utils.seed import seed_everything

import torch
from model.direction_model import *
from model.three_direction_model import *

def inference_loop(dataloader, test_label):
    device = 'cuda'
    epoch = CFG['EPOCHS']

    aweight_save = f"D:\\새 폴더\\8월\\0810\\three_direction\\attention_model\\axial\\axial_{epoch}.pt"
    sweight_save = f"D:\\새 폴더\\8월\\0810\\three_direction\\attention_model\\sagittal\\sagittal_{epoch}.pt"
    cweight_save = f"D:\\새 폴더\\8월\\0810\\three_direction\\attention_model\\coronal\\coronal_{epoch}.pt"
    eweight_save = f"D:\\새 폴더\\8월\\0810\\three_direction\\attention_model\\ensemble\\ensemble_{epoch}.pt"

    size=len(dataloader.dataset)

    seed_everything(CFG['SEED'])

    _amodel=axial_model().cuda()
    _smodel=sagittal_model().cuda()
    _cmodel=coronal_model().cuda()
    emodel=ensemble_model().cuda()

    amodel=nn.DataParallel(_amodel).to(device)
    smodel=nn.DataParallel(_smodel).to(device)
    cmodel=nn.DataParallel(_cmodel).to(device)
    #emodel=nn.DataParallel(_emodel).to(device)

    amodel.load_state_dict(torch.load(aweight_save))
    smodel.load_state_dict(torch.load(sweight_save))
    cmodel.load_state_dict(torch.load(cweight_save))
    emodel.load_state_dict(torch.load(eweight_save))

    pred=[]

    amodel.eval()
    smodel.eval()
    cmodel.eval()
    emodel.eval()

    check=size

    with torch.no_grad():
        for b, (X,y) in enumerate(dataloader):
            X=torch.transpose(X, 1, 0)
            X=torch.reshape(X, (X.shape[0], X.shape[1]*X.shape[2], X.shape[3], X.shape[4], X.shape[5]))
            y=y.to(device).float()

            axial_pred = amodel(X[0].to(device).float())
            sagittal_pred = smodel(X[1].to(device).float())
            coronal_pred = cmodel(X[2].to(device).float())

            person_fc=[]

            # Attention
            if check>=CFG['BATCH_SIZE']:
                for i in range(CFG['BATCH_SIZE']):
                    concat_result1 = torch.concat([axial_pred[i], sagittal_pred[i], coronal_pred[i]], dim=1)
                    concat_result1 = torch.reshape(concat_result1, (3, 20, 2048))

                    person_fc.append(concat_result1)
            else:
                for i in range(check):
                    concat_result1 = torch.concat([axial_pred[i], sagittal_pred[i], coronal_pred[i]], dim=1)
                    concat_result1 = torch.reshape(concat_result1, (3, 20, 2048))

                    person_fc.append(concat_result1)

            check-=CFG['BATCH_SIZE']

            person_fc=torch.stack(person_fc, 0)
            ensemble_pred = emodel(person_fc)   

            for i in ensemble_pred:
                pred.append(i.detach().cpu().tolist()[1])

    roc_auc(test_label, pred)