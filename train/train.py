import torch
from utils.config import CFG
from utils.seed import *

def train_loop(dataloader, amodel, smodel, cmodel, emodel, optimizer, loss_fn, device):
    batchsize = len(dataloader)
    data_size = len(dataloader.dataset)

    seed_everything(CFG['SEED'])
    check=data_size

    print(f"Data set {data_size}")
    losses, correct= 0,0

    for b, (X,y) in enumerate(dataloader):
        X = torch.transpose(X, 1, 0)
        X = torch.reshape(X, (X.shape[0], X.shape[1]*X.shape[2], X.shape[3], X.shape[4], X.shape[5]))
        y = y.to(device).float()

        axial_pred = amodel(X[0].to(device).float())
        sagittal_pred = smodel(X[1].to(device).float())
        coronal_pred = cmodel(X[2].to(device).float())

        person_fc=[]

        #attention
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

        loss = loss_fn(ensemble_pred, y)

        optimizer.zero_grad()

        losses+=loss.item()
        loss.backward()

        optimizer.step()

        correct+=(ensemble_pred.argmax(1)==y.argmax(1)).sum().item()

        if b % 10 == 0:
            print(f"Loss : {loss:>.5f} Correct : [{correct}/{data_size}]  [{b}/{batchsize}]")

    print(f"\nTrain\n Loss : {(losses/batchsize):>.5f} ACC : {(correct/data_size*100):>0.2f} ({correct}/{data_size})")

    return losses/batchsize, correct/data_size*100