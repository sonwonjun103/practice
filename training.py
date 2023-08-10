import time

from train.train import *
from test.test import *
from utils.seed import *

def training(train_dataloader, test_dataloader, amodel, smodel, cmodel, emodel, optimizer, loss_fn, test_label, device):
    train_loss=[]
    train_acc=[]

    val_loss=[]
    val_acc=[]

    sensitivity=[]
    specificity=[]
    AUC=[]
    ACC=[]

    seed_everything(CFG['SEED'])
    train_start=time.time()

    amodel.train()
    smodel.train()
    cmodel.train()
    emodel.train()

    for epoch in range(CFG['EPOCHS']):
        print(f"Start {epoch+1}")
        start=time.time()
        trainloss, trainacc = train_loop(train_dataloader, amodel, smodel, cmodel, emodel, optimizer, loss_fn, device)

        aweight_save=f"D:\\새 폴더\\8월\\0810\\three_direction\\attention_model\\axial\\axial_{epoch+1}.pt"
        torch.save(amodel.state_dict(), aweight_save)

        sweight_save=f"D:\\새 폴더\\8월\\0810\\three_direction\\attention_model\\sagittal\\sagittal_{epoch+1}.pt"
        torch.save(smodel.state_dict(), sweight_save)

        cweight_save=f"D:\\새 폴더\\8월\\0810\\three_direction\\attention_model\\coronal\\coronal_{epoch+1}.pt"
        torch.save(cmodel.state_dict(), cweight_save)

        eweight_save=f"D:\\새 폴더\\8월\\0810\\three_direction\\attention_model\\ensemble\\ensemble_{epoch+1}.pt"
        torch.save(emodel.state_dict(), eweight_save)

        testloss, testacc, sens, spec, auc, acc = test_loop(test_dataloader, loss_fn, test_label, aweight_save, sweight_save, cweight_save, eweight_save, device)
        end=time.time()

        train_loss.append(trainloss)
        train_acc.append(trainacc)

        val_loss.append(testloss)
        val_acc.append(testacc)

        sensitivity.append(sens)
        specificity.append(spec)
        AUC.append(auc)
        ACC.append(acc)

        print(f"End {epoch+1}")

        print(f"Epoch Time : {(end-start)//60} min {((end-start)%60):>0.3f} sec\n")

    train_end=time.time()

    print(f"\nTrain Time : {(train_end-train_start)//60} min {((train_end-train_start)%60):>0.3f} sec")
    return train_loss, train_acc, val_loss, val_acc, sensitivity, specificity, AUC, ACC