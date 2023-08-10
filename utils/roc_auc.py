from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

def roc_auc(label, pred, show):
    fpr, tpr, thresholds = roc_curve(label, pred)
    J=tpr-fpr

    idx=np.argmax(J)
    best_thresh = thresholds[idx]

    sens, spec = tpr[idx], 1-fpr[idx]

    asd, tc = 20,20

    acc=(sens*20 + spec*20) / 40
    auc = roc_auc_score(label, pred)

    if show==1:
        # plot the roc curve for the model
        plt.title("ROC CURVE")
        plt.plot([0,1], [0,1], linestyle='--', markersize=0.01, color='black')
        plt.plot(fpr, tpr, marker='.', color='black', markersize=0.05)
        plt.scatter(fpr[idx], tpr[idx], marker='+', s=100, color='r',
                    label = 'Best threshold = %.3f, \nSensitivity : %.3f (%d / %d), \nSpecificity = %.3f (%d / %d), \nAUC = %.3f , \nACC = %.3f (%d / %d)' % (best_thresh, sens, (sens*asd), asd, spec, (spec*tc), tc, auc, acc, (sens*asd+spec*tc), 40))
        plt.legend()

    return sens, spec, auc, acc, best_thresh
