from datasets import *
from winclip import *
from params import state_level, template_level, datasets, DS_DIR
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import sys

def calculate_pro_score(scores, gt, expected=0.3, steps=200):
    maxScore = scores.max()
    minScore = scores.min()
    delta = (maxScore - minScore) / steps
    all_pro = []
    all_fpr = []
    bin_map = np.zeros_like(scores, dtype=bool)
    for step in steps:
        pro = []
        fpr = []
        thresh = minScore + step * delta
        bin_map[scores > thresh] = 1
        bin_map[scores <= thresh] = 0

        for i in range(len(scores)):
            label_map = measure.label(gt[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = scores[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)

        all_pro.append(np.array(pro).mean())
        masks_neg = ~gt
        fpr = np.logical_and(masks_neg, bin_map).sum() / masks_neg.sum()
        all_fpr.append(fpr)

    all_pro = np.array(all_pro)
    all_fpr = np.array(all_fpr)
    indices = all_fpr <= expected
    c_fpr = all_fpr[indices]
    c_pro = all_pro[indices]
    c_fpr = (c_fpr - c_fpr.min()) / (c_fpr.max() - c_fpr.min())
    pro_score = auc(c_fpr, c_pro)
    return pro_score

def eval(shots=0, mode='AC', dataset='mvtec-ad'):
    win_model = WinCLIP(state_level, template_level, shots=shots, option=mode)
    win_model.eval = True
    win_model.to(DEVICE)

    obj_list = datasets[dataset]
    with torch.no_grad():
        if mode == 'AC':
            all_auroc = []
            all_aupr = []
            all_f1 = []
            for object_name in obj_list:
                score_list = []
                gt_list = []
                ds = init_dataset(dataset, object_name, shot=shots, preprocess=win_model.preprocess)
                dataloader = DataLoader(ds, batch_size=1, num_workers=1, shuffle=False)
                for data in tqdm(dataloader, desc="Running: "):
                    ref_list, img, isAbno, indice, gt, sizes = data
                    score = win_model.forward(object_name, img, ref_list, shot=shots, option=mode, out_size=sizes)
                    score_list.append(score)
                    gt_list.append(isAbno[0].numpy())
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
        else:
            all_auroc = []
            all_f1 = []
            all_pro = []
            for object_name in obj_list:
                score_list = []
                gt_list = []
                ds = init_dataset(dataset, object_name, shot=shots, preprocess=win_model.preprocess)
                dataloader = DataLoader(ds, batch_size=1, num_workers=1, shuffle=False)
                for data in tqdm(dataloader, desc="Running: "):
                    ref_list, img, isAbno, indice, gt, sizes = data
                    score = win_model.forward(object_name, img, ref_list, shot=shots, option=mode, out_size=sizes)
                    score_list.append(score)
                    gt = gt.squeeze(0).numpy()
                    gt[gt > 0] = 1
                    gt_list.append(gt)
                
                scores = np.array(score_list)
                gt = np.asarray(gt_list, dtype=int)
                p_auroc = roc_auc_score(gt.flatten(), scores.flatten())
                all_auroc.append(p_auroc)
                precision, recall, _ = precision_recall_curve(gt.flatten(), scores.flatten())
                up = 2 * precision * recall
                down = precision + recall
                f1_score = np.divide(up, down, out=np.zeros_like(up), where=down != 0)
                index = np.argmax(f1_score)
                f1_max = f1_score[index]
                all_f1.append(f1_max)
                pro = calculate_pro_score(np.array(score_list), np.array(gt_list))
                all_pro.append(pro)
                print("Obj Type: {}, p_AUROC={:.1f}, PRO_AUC={:1.f}, p_F1_MAX={:.1f}".format(object_name, p_auroc * 100.0, pro * 100.0, f1_max * 100.0))

            print("AVG_p_AUROC: {:.1f}, AVG_PRO_AUC: {:.1f}, AVG_p_F1_MAX: {:.1f}".format(np.mean(all_auroc) * 100.0, np.mean(all_pro) * 100.0, np.mean(all_f1) * 100.0))

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