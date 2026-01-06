from datasets import *
from winclip import *
from params import state_level, template_level, DS_DIR
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import sys

#print a segmentation view
def segment(anomaly_map, img, normalized=True, isDefected=True, save_path=None):
    image = img
    if normalized:
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std = torch.tensor(STD).view(3, 1, 1)
        image = img * std + mean
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0)
    plt.figure(figsize=(6,6))
    plt.title("Segmentation")
    plt.imshow(image)
    plt.imshow(anomaly_map, cmap='jet', alpha=0.5)
    plt.colorbar(label='Anomaly Score')
    plt.axis("off")
    if save_path: 
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

@torch.no_grad()
def run(object_name, dataset='mvtec-ad', shots=0, dir_path=None, num=5):
    model = WinCLIP(state_level, template_level, shots=shots, option='AS').to(DEVICE)
    ds = init_dataset(dataset, object_name, shot=shots, preprocess=model.preprocess)
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    count = 0
    for data in tqdm(loader, desc="running"):
        if count >= num:
            break
        ref_list, img, isAbno, indice, gt = data
        if isAbno.item() == 0:
            continue
        score = model(object_name, img, ref_list, shot=shots, option="AS")
        save_file = os.path.join(dir_path, f"seg_{indice.item()}.png") if dir_path is not None else None
        segment(score, img.squeeze(0), isDefected=True, save_path=save_file)
        if save_file is not None:
            mask_file = save_file.replace("seg_", "mask_")
            plt.figure(figsize=(6,6))
            plt.title("Mask")
            plt.imshow(gt.squeeze(), cmap='gray')
            plt.axis("off")
            plt.savefig(mask_file, bbox_inches="tight", dpi=300)
            plt.close()
        if save_file is None:
            break
        count += 1


if __name__ == "__main__":
    run("screw", shots=1, dir_path="../Results/MVtec-AD/screw_1", num=15)
    # run("cashew", shots=1, dir_path="../Results/VisA/cashew_1", num=10, dataset='visa')