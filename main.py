from datasets import *
from winclip import *
from params import state_level, template_level, datasets, DS_DIR
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

import sys

def eval(shots=0, mode='AC', dataset='mvtec-ad'):
    win_model = WinCLIP(state_level, template_level, shots=shots, option=mode)
    win_model.to(DEVICE)

    all_auroc = []
    all_aupr = []
    all_f1 = []
    all_score = []
    all_gt = []
    obj_list = datasets[dataset]
    with torch.no_grad():
        for object_name in obj_list:
            score_list = []
            gt_list = []
            mvtec_adds = MVtecADDataset(object_name, DS_DIR[0], shot=shots, preprocess=win_model.preprocess)
            dataloader = DataLoader(mvtec_adds, batch_size=1, num_workers=1, shuffle=False)
            for data in tqdm(dataloader, desc="Running: "):
                ref_list, img, isAbno, indice, gt = data
                score = win_model.forward(object_name, img, ref_list, shot=shots, option=mode)
                if mode == 'AC':
                    score_list.append(score)
                    gt_list.append(isAbno[0].numpy())
                    all_score.append(score)
                    all_gt.append(isAbno[0].numpy())
                else:
                    score_list.extend(score.reshape(-1))
                    gt_list.extend(gt.reshape(-1).numpy())
                    all_score.extend(score.reshape(-1))
                    all_gt.extend(gt.reshape(-1).numpy())

            auroc = roc_auc_score(gt_list, score_list)
            precision, recall, _ = precision_recall_curve(gt_list, score_list)
            aupr = auc(recall, precision)
            f1_max = 0
            for threshold in np.arange(0, 1, 0.01):
                y_pred = (score_list > threshold).astype(int)
                f1 = f1_score(gt_list, y_pred)
                if f1 > f1_max:
                    f1_max = f1
            print("Obj Type: {}, AUROC={:.1f}, AUPR={:.1f}, F1_MAX={:.1f}".format(object_name, auroc * 100.0, aupr * 100.0, f1_max * 100.0))
            all_auroc.append(auroc)
            all_aupr.append(aupr)
            all_f1.append(f1_max)


    print("AVG_AUROC: {:.1f}, AVG_AUPR: {:.1f}, AVG_F1_MAX: {:.1f}".format(np.mean(all_auroc) * 100.0, np.mean(all_aupr) * 100.0, np.mean(all_f1) * 100.0))
    precision, recall, _ = precision_recall_curve(all_gt, all_score)
    aupr = auc(recall, precision)
    f1_max = 0
    for threshold in np.arange(0, 1, 0.01):
        y_pred = (all_score > threshold).astype(int)
        f1 = f1_score(all_gt, y_pred)
        if f1 > f1_max:
            f1_max = f1
    print("All AUROC: {:.1f}, All AUPR: {:.1f}, All F1_MAX: {:.1f}".format(roc_auc_score(all_gt, all_score) * 100.0, aupr * 100.0, f1_max * 100.0))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if len(sys.argv) > 3:
            raise ValueError("Too many argument passed, expect 1-2")
        flag  = 'AC' #default
        shots = 0    #default
        for arg in sys.argv[1:]:
            if arg.startswith("-"):
                flag = arg.lstrip("-")
            else:
                try:
                    shots = int(arg)
                except ValueError:
                    raise ValueError("Invalid argument")
        eval(shots=shots, mode=flag)
    else:
        eval()