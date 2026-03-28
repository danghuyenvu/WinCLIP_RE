from datasets import *
from winclip import *
from params import state_level, template_level, DS_DIR
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import os, sys, argparse, json

#print a segmentation view
def segment(anomaly_map, img, normalized=True, save_path=None):
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

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # Instead of printing the error message, just show help
        self.print_help(sys.stderr)
        sys.exit(2)

def main():
    global DEVICE
    parser = CustomArgumentParser(description="Run WinCLIP anomaly segmentation on user images")
    parser.add_argument("-i", "--image", required=True, help="Path to target image")
    parser.add_argument("-r", "--result", required=True, help="Path to result folder")
    parser.add_argument("-s", "--shots", type=int, default=0, help="Number of shots (default: 0 for zero-shot)")
    parser.add_argument("-re", "--references", help="Path to reference folder (required if shots > 0)")
    parser.add_argument("-o", "--object-name", required=True, help="Name of the object")
    parser.add_argument("-p", "--prompts", help="Path to custom prompts file (optional)")
    parser.add_argument("-d", "--device", help="Device name to use (cuda, mps, cpu, default = cpu/mps)")
    args = parser.parse_args()

    # Load prompts if provided
    state_level_local, template_level_local = state_level, template_level
    if args.prompts:
        with open(args.prompts, "r") as f:
            prompts = json.load(f)
        state_level_local = prompts["state_level"]
        template_level_local = prompts["template_level"]

    if args.device:
        DEVICE = args.device
    model = WinCLIP(state_level_local, template_level_local, shots=args.shots, option="AS").to(DEVICE)

    ds = UserDataset(args.object_name, data_dir=args.image, shot=args.shots, preprocess=model.preprocess, reference_dir=args.references)

    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    for data in tqdm(loader, desc="running"):
        ref_list, img, _, indice, _ = data
        score = model(args.object_name, img, ref_list, shot=args.shots, option="AS")
        save_file = os.path.join(args.result, f"seg_{indice.item()}.png")
        segment(score, img.squeeze(0), save_path=save_file)

    print(f"Segmentation result saved to {args.result}")

if __name__ == "__main__":
    main()