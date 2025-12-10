from datasets import *
from winclip import *
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

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
	model = WinCLIP(state_level, template_level).to(DEVICE)
	ds = ADDataset("bottle", DS_DIR, shot=2, preprocess=model.preprocess)
	ref_list, img, isAbno, indice, image_path = ds[15]
	ref_list = [x.unsqueeze(0) for x in ref_list]
	anomaly_map = model.forward("bottle", img.unsqueeze(0), shot=2, option="AS", ref_list=ref_list)
	# binary_mask = anomaly_map > 0.33
	# segment(anomaly_map, img)
	plot(anomaly_map, img)

def eval(shots=0, mode='all'):
	win_model = WinCLIP(state_level, template_level)
	win_model.to(DEVICE)

	all_auroc = []
	all_aupr = []
	all_f1 = []
	all_score = []
	all_gt = []
	with torch.no_grad():
		for object_name in OBJECT_TYPE:
			score_list = []
			gt_list = []
			mvtec_adds = ADDataset(object_name, DS_DIR, shot=shots, preprocess=win_model.preprocess)
			dataloader = DataLoader(mvtec_adds, batch_size=1, num_workers=1, shuffle=False)
			for data in tqdm(dataloader, desc="Running: "):
				ref_list, img, isAbno, indice, img_path = data
				score = win_model.forward(object_name, img, ref_list, shot=shots, option=mode)
				score_list.append(score)
				gt_list.append(isAbno[0].numpy())
				all_score.append(score)
				all_gt.append(isAbno[0].numpy())

			auroc = roc_auc_score(gt_list, score_list)
			precision, recall, _ = precision_recall_curve(gt_list, score_list)
			aupr = auc(recall, precision)
			f1_max = 0
			for threshold in np.arange(0, 1, 0.01):
				y_pred = (score_list > threshold).astype(int)
				f1 = f1_score(gt_list, y_pred)
				if f1 > f1_max:
					f1_max = f1
			print("Obj Type: {}, AUROC={}, AUPR={}, F1_MAX={}".format(object_name, auroc, aupr, f1_max))
			all_auroc.append(auroc)
			all_aupr.append(aupr)
			all_f1.append(f1_max)

	print("AVG_AUROC: {}, AVG_AUPR: {}, AVG_F1_MAX: {}".format(np.mean(all_auroc), np.mean(all_aupr), np.mean(all_f1)))
	precision, recall, _ = precision_recall_curve(all_gt, all_score)
	aupr = auc(recall, precision)
	f1_max = 0
	for threshold in np.arange(0, 1, 0.01):
		y_pred = (all_score > threshold).astype(int)
		f1 = f1_score(all_gt, y_pred)
		if f1 > f1_max:
			f1_max = f1
	print("All AUROC: {}, All AUPR: {}, All F1_MAX: {}".format(roc_auc_score(all_gt, all_score), aupr, f1_max))


if __name__ == "__main__":
    # eval()
	run()