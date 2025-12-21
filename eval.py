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
    win_model.eval = True
    win_model.to(DEVICE)

    all_auroc = []
    all_aupr = []
    all_f1 = []
    obj_list = datasets[dataset]
    with torch.no_grad():
        for object_name in obj_list:
            score_list = []
            gt_list = []
            ds = init_dataset(dataset, object_name, shot=shots, preprocess=win_model.preprocess)
            dataloader = DataLoader(ds, batch_size=1, num_workers=1, shuffle=False)
            for data in tqdm(dataloader, desc="Running: "):
                ref_list, img, isAbno, indice, gt, sizes = data
                score = win_model.forward(object_name, img, ref_list, shot=shots, option=mode, out_size=sizes)
                if mode == 'AC':
                    score_list.append(score)
                    gt_list.append(isAbno[0].numpy())
                else:
                    score_list.extend(score)
                    gt_list.extend(np.array(gt.squeeze(0))) #get rid of batch dimension

            auroc = roc_auc_score(gt_list, score_list)
            precision, recall, _ = precision_recall_curve(gt_list, score_list)
            aupr = auc(recall, precision)
            up = 2 * precision * recall
            down = precision + recall
            f1_score = np.divide(up, down, out=np.zeros_like(up), where=down != 0)
            max_index = np.argmax(f1_score)
            f1_max = f1_score[max_index]
            print("Obj Type: {}, AUROC={:.1f}, AUPR={:.1f}, F1_MAX={:.1f}".format(object_name, auroc * 100.0, aupr * 100.0, f1_max * 100.0))
            all_auroc.append(auroc)
            all_aupr.append(aupr)
            all_f1.append(f1_max)

    print("AVG_AUROC: {:.1f}, AVG_AUPR: {:.1f}, AVG_F1_MAX: {:.1f}".format(np.mean(all_auroc) * 100.0, np.mean(all_aupr) * 100.0, np.mean(all_f1) * 100.0))

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