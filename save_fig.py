import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def save_fig(info, info_type):
    if info_type==1:
        type = 'train_loss'
        plt.figure(figsize=(15, 7))
        plt.plot(info, label=type)
        plt.title(type)
        plt.legend()
        plt.savefig(f"D:\\새 폴더\\8월\\0810\\main\\{type}.png")
    
    elif info_type==2:
        type = 'train_acc'
        plt.figure(figsize=(15, 7))
        plt.plot(info, label=type)
        plt.title(type)
        plt.legend()
        plt.savefig(f"D:\\새 폴더\\8월\\0810\\main\\{type}.png")

    elif info_type==2:
        type = 'test_loss'
        plt.figure(figsize=(15, 7))
        plt.plot(info, label=type)
        plt.title(type)
        plt.legend()
        plt.savefig(f"D:\\새 폴더\\8월\\0810\\main\\{type}.png")

    elif info_type==3:
        type = 'test_acc'
        plt.figure(figsize=(15, 7))
        plt.plot(info, label=type)
        plt.title(type)
        plt.legend()
        plt.savefig(f"D:\\새 폴더\\8월\\0810\\main\\{type}.png")

    elif info_type==4:
        type = 'sensitivity'
        plt.figure(figsize=(15, 7))
        plt.plot(info, label=type)
        plt.title(type)
        plt.legend()
        plt.savefig(f"D:\\새 폴더\\8월\\0810\\main\\{type}.png")

    elif info_type==5:
        type = 'specificity'
        plt.figure(figsize=(15, 7))
        plt.plot(info, label=type)
        plt.title(type)
        plt.legend()
        plt.savefig(f"D:\\새 폴더\\8월\\0810\\main\\{type}.png")

    elif info_type==6:
        type = 'ACC'
        plt.figure(figsize=(15, 7))
        plt.plot(info, label=type)
        plt.title(type)
        plt.legend()
        plt.savefig(f"D:\\새 폴더\\8월\\0810\\main\\{type}.png")

def roc_auc(label, pred):
    fpr, tpr, thresholds = roc_curve(label, pred)
    J=tpr-fpr

    idx=np.argmax(J)
    best_thresh = thresholds[idx]

    sens, spec = tpr[idx], 1-fpr[idx]

    asd, tc = 20,20

    acc=(sens*20 + spec*20) / 40
    auc = roc_auc_score(label, pred)

    # plot the roc curve for the model
    plt.title("ROC CURVE")
    plt.plot([0,1], [0,1], linestyle='--', markersize=0.01, color='black')
    plt.plot(fpr, tpr, marker='.', color='black', markersize=0.05)
    plt.scatter(fpr[idx], tpr[idx], marker='+', s=100, color='r',
                label = 'Best threshold = %.3f, \nSensitivity : %.3f (%d / %d), \nSpecificity = %.3f (%d / %d), \nAUC = %.3f , \nACC = %.3f (%d / %d)' % (best_thresh, sens, (sens*asd), asd, spec, (spec*tc), tc, auc, acc, (sens*asd+spec*tc), 40))
    plt.legend()

    plt.savefig(f"D:\\새 폴더\\8월\\0810\\main\\roc_curve.png")