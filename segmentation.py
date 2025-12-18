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
def segment(anomaly_map, img, normalized=True, isDefected=True):
    print("State: Anomaly" if isDefected else "State: Normal")
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
        plt.show()
    else:
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0)
        plt.figure(figsize=(6,6))
        plt.title("Segmentation")
        plt.imshow(image)
        plt.imshow(anomaly_map, cmap='jet', alpha=0.5)
        plt.colorbar(label='Anomaly Score')
        plt.axis("off")
        plt.show()

def plot(anomaly_map, img, normalized=True):
    if normalized:
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std = torch.tensor(STD).view(3, 1, 1)
        image = img * std + mean
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0)

        plt.figure(figsize=(6,6))
        plt.title("Anomaly map")
        plt.imshow(anomaly_map, cmap='jet')
        plt.colorbar(label='Anomaly Score')
        plt.axis("off")

        plt.figure(figsize=(6,6))
        plt.title("Image")
        plt.imshow(image)
        plt.colorbar(label='Haha')
        plt.axis("off")
        plt.show()
    else:
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0)
        plt.figure(figsize=(6,6))
        plt.title("Anomaly map")
        plt.imshow(anomaly_map, cmap='jet')
        plt.colorbar(label='Anomaly Score')
        plt.axis("off")

        plt.figure(figsize=(6,6))
        plt.title("Image")
        plt.imshow(image)
        plt.colorbar(label='Haha')
        plt.axis("off")
        plt.show()

def run():
    model = WinCLIP(state_level, template_level, shots=2, option='AS').to(DEVICE)
    ds = MVtecADDataset("bottle", DS_DIR[0], shot=2, preprocess=model.preprocess)
    ref_list, img, isAbno, indice, gt = ds[15]
    ref_list = [x.unsqueeze(0) for x in ref_list]
    anomaly_map = model.forward("bottle", img.unsqueeze(0), shot=0, option="AS", ref_list=ref_list)
    plot(anomaly_map, img)

if __name__ == "__main__":
    run()