import torch
import random
import glob
import os
from random import sample
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

import open_clip

#constants parameters
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
MODEL_NAME = "ViT-B-16-plus-240"
PRETRAINED = "laion400m_e32"
RE_SIZE = 240
DS_NAME = "mvtec-ad"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

#setup states and template text for compositional promt ensemble
state_level = {
	"normal":["{}", "flawless {}", "perfect {}", "unblemished {}",
						"{} without flaw", "{} without defect", "{} without damage"],
	"anomaly":["damaged {}", "{} with flaw", "{} with defect", "{} with damage"]
}
template_level = [
	"a cropped photo of the {}.",
	"a cropped photo of a {}.",
	"a close-up photo of a {}.",
	"a close-up photo of the {}.",
	"a bright photo of a {}.",
	"a bright photo of the {}.",
	"a dark photo of a {}.",
	"a dark photo of the {}.",
	"a jpeg corrupted photo of a {}.",
	"a jpeg corrupted photo of the {}.",
	"a blurry photo of the {}.",
	"a blurry photo of a {}.",
	"a photo of the {}.",
	"a photo of a {}.",
	"a photo of a small {}.",
	"a photo of the small {}.",
	"a photo of a large {}.",
	"a photo of the large {}.",
	"a photo of a {} for visual inspection.",
	"a photo of the {} for visual inspection.",
	"a photo of a {} for anomaly detection.",
	"a photo of the {} for anomaly detection."
]
VISA_OBJ = [
	"candle",
	"capsules",
	"cashew",
	"chewinggum",
	"fryum",
	"macaroni1",
	"macaroni2",
	"pcb1",
	"pcb2",
	"pcb3",
	"pcb4",
	"pipe_fryum",
	'all'
]	
MVTEC_AD_OBJ = [
	"bottle",
	"cable",
	"capsule",
	"carpet",
	"grid",
	"hazelnut",
	"leather",
	"metal_nut",
	"pill",
	"screw",
	"tile",
	"toothbrush",
	"transistor",
	"wood",
	"zipper"
]

OBJECT_TYPE = VISA_OBJ if DS_NAME == "visa" else MVTEC_AD_OBJ

DS_DIR = "../Datasets/MVtecAD"

class ADDataset(Dataset):
	def __init__(self, obj_type, data_dir, spatial_size=240, mode="train", shot=0, preprocess=None):
		super(ADDataset, self).__init__()
		self.shot = shot
		self.object_type = obj_type
		self.data_dir = data_dir
		self.spatial_size = spatial_size
		self.mode = mode
		self.preprocess = preprocess
		self.pre_transform = transforms.Compose([
			transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
			transforms.CenterCrop(size=(240, 240)),
			transforms.Lambda(self.convert_rgb),
			transforms.ToTensor(),
			transforms.Normalize(mean=MEAN, std=STD)
			])
		if self.preprocess is not None:
			self.preprocess.transforms[0] = transforms.Resize(
				size=(240, 240),
				interpolation=transforms.InterpolationMode.BICUBIC,
				max_size=None, antialias=None
			)
			self.preprocess.transforms[1] = transforms.CenterCrop(size=(240, 240))
		self.init_dataset()

	def init_dataset(self):
		self.img_path = []
		if self.object_type == "all":
			#get all images path of all objects
			for x in OBJECT_TYPE:
				cur_obj = os.path.join(self.data_dir, x)
				cur_obj = os.path.join(cur_obj, "test")
				self.img_path.extend(glob.glob(os.path.join(cur_obj, "**", "*.*")))
			self.img_path = sorted(self.img_path)
			return

		type_dir = os.path.join(self.data_dir, self.object_type)
		test_dir = os.path.join(type_dir, "test")

		#get all images paths in test folder
		self.img_path.extend(glob.glob(os.path.join(test_dir, "**", "*.*")))

	def convert_rgb(self, x):
		return x.convert('RGB')

	def __getitem__(self, indice):
		ref_list = []
		img = None
		if self.shot != 0:
			normal_image = []
			# get shot number of normal images for reference
			if self.object_type == "all":
				raise ValueError("Only all for Zero shot")

			train_dir = os.path.join(self.data_dir, self.object_type, "train", "good")
			normal_image.extend(glob.glob(os.path.join(train_dir, "*.*")))
			normal_image = np.random.RandomState(10).choice(normal_image, self.shot)
			for x in normal_image:
				if self.preprocess is not None:
					ref_list.append(self.transform_image(x, self.preprocess))
				else:
					ref_list.append(self.transform_image(x, self.pre_transform))

		if indice >= len(self.img_path):
			raise ValueError("Invalid indice")
		image_path = self.img_path[indice]
		folder_path, image = os.path.split(image_path)
		isAbno = np.array([1], dtype=np.float32)
		if os.path.basename(folder_path) == "good":
			isAbno = np.array([0], dtype=np.float32)
		img = self.transform_image(image_path, self.preprocess) if self.preprocess is not None else self.transform_image(image_path, self.pre_transform)

		return ref_list, img, isAbno, indice, image_path

	def __len__(self):
		return len(self.img_path)

	def transform_image(self, image, preprocess=None):
		"""
		do pretransform for image
		"""
		image = Image.open(image)
		height, width = image.size
		if height == width:
			processed = preprocess(image)
			return processed
		else:
			cropped_image = self.crop_image(image)
			processed = []
			for i in cropped_image:
				processed.append(preprocess(i))
			return processed

	def crop_image(self, image, stride_ratio=0.8):
		"""
		image tiling: crop images into list of squared images based on smaller size
		"""
		width, height = image.size
		larger = max(height, width)
		smaller = min(height, width)
		stride = int(stride_ratio * smaller)
		cropped_list = []

		if smaller == height:
			#slide horizontally
			for x in range(0, larger, stride):
				if x + smaller >= larger:
					cropped_list.append(image.crop((larger - smaller, 0, larger, smaller)))
				else:
					cropped_list.append(image.crop((x, 0, x + smaller, smaller)))
		else:
			#slide vertically
			for x in range(0, larger, stride):
				if x + smaller >= larger:
					cropped_list.append(image.crop((0, larger - smaller, smaller, larger)))
				else:
					cropped_list.append(image.crop((0, x, smaller, x + smaller)))

		return cropped_list


class WinCLIP(nn.Module):
	def __init__(self, states, templates):
		super().__init__()
		self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained="winCLIP_ViT_B16P_laion400m.pt")
		# self.model.load_state_dict(torch.load("winCLIP_ViT_B16P_laion400m.pt", map_location=DEVICE))
		self.model.eval()
		self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
		self.window_mask = []
		self.visual = self.model.visual
		self.reference_bank = []

		self.state_level = states
		self.template_level = templates

		#set output_tokens to true to get output tokens instead of pooled cls
		# self.visual.output_tokens = True

	@torch.no_grad()
	def encode_image(self, img, windowmask=None, normalize=False):
		"""
		WinCLIP image encoder performing both normal image encoding and 
		window-scale encoding by dropping out masked patches before feeding to the CLIP-encoder
		
		:param self: Description
		:param img: query image
		:param windowmask: mask
		:param normalize: normalization for result tokens
		"""
		if windowmask is not None:
			x = self.visual.conv1(img)
			x = x.reshape(x.shape[0], x.shape[1], -1)
			x = x.permute(0, 2, 1)

			#concat cls and add positional embedding
			x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
			x = x + self.visual.positional_embedding.to(x.dtype)
			window_cls_list = []
			for mask in windowmask:
				#perform window masking by selecting patches to pass in transformer encoder
				scaled_x = []
				mask = mask.T #now has shape (num_mask, mask_size)
				num_mask, L = mask.shape
				class_index = torch.zeros((mask.shape[0], 1), dtype=torch.int32).to(mask)
				mask = torch.cat((class_index, mask.int()), dim=1)
				for i in mask:
					mx = torch.index_select(x, 1, i.int())
					scaled_x.append(torch.index_select(x, 1, i.int()))
				mx = torch.cat(scaled_x)
				mx = self.visual.patch_dropout(mx)
				mx = self.visual.ln_pre(mx)
				mx = self.visual.transformer(mx)

				cls = mx[:, 0]
				cls = self.visual.ln_post(cls)

				if self.visual.proj is not None:
					cls = cls @ self.visual.proj

				cls = F.normalize(cls, dim=-1)

				# window_cls_list.append(cls.reshape((num_mask, 1, 1, -1)).permute(1, 0, 2, 3))
				window_cls_list.append(cls)

			#continue passing into transformer encoder the whole image
			x = self.visual.patch_dropout(x)
			x = self.visual.ln_pre(x)
			x = self.visual.transformer(x)
			cls = x[:, 0]
			tokens = x[:, 1:]
			cls = self.visual.ln_post(cls)
			if self.visual.proj is not None:
				cls = cls @ self.visual.proj

			cls = F.normalize(cls, dim=-1)

			return window_cls_list, tokens, cls
		else:
			#getting the image scale feature then just get the cls token
			features = self.model.encode_image(img)
			return F.normalize(features, dim=-1) if normalize else features

	@torch.no_grad()
	def encode_text(self, object_name='all'):
		"""
		Composition Promt Ensemble generate based on object_name
		
		:param self: Doodoo
		:param object_name: name of the query object
		"""
		normal_states = [s.format(object_name) for s in self.state_level["normal"]]
		anomaly_states = [s.format(object_name) for s in self.state_level["anomaly"]]

		normal_texts = [t.format(state) for state in normal_states for t in self.template_level]
		anomaly_texts = [t.format(state) for state in anomaly_states for t in self.template_level]

		normal_texts_f = self.tokenizer(normal_texts).to(DEVICE)
		abno_texts_f = self.tokenizer(anomaly_texts).to(DEVICE)
		normal_texts_f = self.model.encode_text(normal_texts_f)
		abno_texts_f = self.model.encode_text(abno_texts_f)

		normal_texts_f /= normal_texts_f.norm(dim=-1, keepdim=True)
		abno_texts_f /= abno_texts_f.norm(dim=-1, keepdim=True)

		normal_texts_f = torch.mean(normal_texts_f, dim=0, keepdim=True)
		abno_texts_f = torch.mean(abno_texts_f, dim=0, keepdim=True)

		normal_texts_f /= normal_texts_f.norm(dim=-1, keepdim=True)
		abno_texts_f /= abno_texts_f.norm(dim=-1, keepdim=True)

		text_features = torch.cat([normal_texts_f, abno_texts_f], dim=0)

		return text_features

	def gen_window_mask(self, patch_size=16, kernel_size=16, stride=16):
		"""generate window scale mask with kernel size"""
		height = 240 // patch_size
		width = 240 // patch_size
		kernel_size = kernel_size // patch_size #calculate window size on patch scale
		stride_size = stride // patch_size
		tmp = torch.arange(1, height*width+1, dtype=torch.float32).reshape(1, 1, height, width)

		mask = torch.nn.functional.unfold(tmp, kernel_size, stride=stride_size)
		return mask
	
	def gen_reference_bank(self, ref_list, masks=None):
		"""
		Generates reference banks for few-shot AC/AS, this includes passing
		normal reference images through WinCLIP encoder results in lists of 
		window/image features for each scale

		So based on what's written in the paper, i supposed the reference bank's
		gonna consists of: 
		- For window-scales is gonna be the features which is noted to be the cls tokens
		- For image scale then it's gonna be the patch tokens, since this is used for image 
		guided scoring, the patch score still usefull cause its context is enriched thanks to
		self-attention mechanism
		
		:param self: Doodoo
		:param ref_list: list of reference images
		:param masks: window mask, if none is passed then reference bank only
		includes normal image features
		"""
		print("Begin generating reference bank...")

		if masks is None:
			#then expects the window_cls to be empty
			for image in ref_list:
				window_cls, tokens, image_cls = self.encode_image(image, masks)
				#current tokens shape of (B, N, D), since B= 1 then just squeeze it
				token_list = []
				tokens = tokens.squeeze()
				for token in tokens:
					token_list.append(token)

				self.reference_bank.append(token_list)
		else:
			for image in ref_list:
				window_cls, tokens, image_cls = self.encode_image(image, masks)

				for cls in window_cls:
					cur_scale = [x for x in cls]
					self.reference_bank.append(cur_scale)

				token_list = []
				tokens = tokens.squeeze()
				for token in tokens:
					token_list.append(token)

				self.reference_bank.append(token_list)

		print("Reference bank generating finished.")

	def calculate_text_anomaly_score(self, text_features, image_features, normal=False):
		"""
		Language-guided anomaly scoring function
		
		:param self: Description
		:param text_features: Text features extracted from Composition Promt Ensemble
		:param image_features: Image features :)
		:param normal: wanna get anomaly score or normality score?
		"""
		if isinstance(image_features, list):
			pass
			# for x in image:
		else:
			if image_features.shape[0] != 1:
				image_features = image_features.unsqueeze(0)
			score = (100.0 * image_features @ text_features.T).softmax(dim=-1)
			if not normal:
				score = score[:, 1]
			else:
				score = score[:, 0]
			return score

	def calculate_visual_anomaly_score(self, patch_feature, feature=None, window_masks=None):
		"""
		Reference association module for given feature
		return: the visual-guided anomaly score for the query image and the visual-guided anomaly score map

		note: if i'm correct the the visual-guided score for each scale is by taking the highest anomaly score
		from each tokens right since it has the highest possibility for that token to be faulty (cannot find the part
		where they get those Mws or Mwm from)
		
		:param self: Description
		:param feature: a batch of window-scale feature
		:param patch_feature: penultimate feature map (including patch tokens at the end of the encoding phase)
		since the cls token of the whole image has gone through a projection layer after returning, it's no more in the
		same dimension with the reference image's patch tokens, so i think using the patch tokens of the query image directly
		might be a better choice
		:param window_masks: window mask
		"""
		# get the list of reference tokens
		# per_scale_AC = []
		anomaly_score_map = []
		if feature is not None:
			#calculate score for each non-image scale features
			for index, (scale, window) in enumerate(zip(feature, window_masks)):
				cur_reference_bank = self.reference_bank[index]
				cur_reference_bank = torch.stack(cur_reference_bank, dim=0)

				dot_product = scale@cur_reference_bank.T
				scale_score = 0.5 * (1.0 - dot_product)
				all_anomaly_score, _ = scale_score.min(dim=1)
				# prediction_score = all_anomaly_score.max()
				# per_scale_AC.append(prediction_score)

				per_scale_map = torch.zeros(15 * 15, device=DEVICE)
				per_scale_weight = torch.zeros(15 * 15, device=DEVICE)
				for x, score in zip(window, all_anomaly_score):
					x = x.long() - 1
					per_scale_map[x] += score
					per_scale_weight += 1.0
				
				per_scale_map = per_scale_map / per_scale_weight
				anomaly_score_map.append(per_scale_map)
		

		#calculate association score for image scale feature
		cur_reference_bank = self.reference_bank[-1]
		cur_reference_bank = torch.stack(cur_reference_bank, dim=0)
		patch_feature = F.normalize(patch_feature, dim=-1)        # (num_patches, D)
		cur_reference_bank = F.normalize(cur_reference_bank, dim=-1)  # (num_refs, D)
		dot_product = patch_feature@cur_reference_bank.T
		image_score = 0.5 * (1.0 - dot_product)
		image_score, _ = image_score.min(dim=1)

		# per_scale_AC.append(image_score.max())
		# AC_score = torch.stack(per_scale_AC, dim=0)
		# AC_score = torch.mean(AC_score, dim=0)

		anomaly_score_map.append(image_score.squeeze())
		anomaly_map = torch.stack(anomaly_score_map, dim=0)
		anomaly_map = torch.mean(anomaly_map, dim=0)
		return anomaly_map
	

	@torch.no_grad()
	def forward(self, object_name, image, ref_list, shot=0, option="AC"):
		"""
		Main pipeline of the model
		
		:param self: Doodoo
		:param object_name: name of the object needed to generate CPE
		:param image: query image
		:param ref_list: list of reference images used for few-shots
		:param shot: number of shot, auto=0 shot
		:param option: 'AC' or 'AS' or None to run all evaluate metrics for both AS and AC (to be updated)
		"""
		with torch.no_grad():
			text_features = self.encode_text(object_name)
			image = image.to(DEVICE)
			ref_list = [x.to(DEVICE) for x in ref_list]
			if option=="AC":
				if shot == 0:
					#WinCLIP zero-shot AC basically applying CPE to CLIP image encoder and text encoder
					image_features = self.encode_image(image)
					return self.calculate_text_anomaly_score(text_features, image_features).detach().cpu().numpy()
				else:
					#adding aggregated vision-based anomaly score
					window_masks_1 = self.gen_window_mask(kernel_size=32).squeeze().to(DEVICE)
					window_masks_2 = self.gen_window_mask(kernel_size=48).squeeze().to(DEVICE)
					window_masks = [window_masks_1, window_masks_2]
					self.gen_reference_bank(ref_list, window_masks)
					window_cls, tokens, image_cls = self.encode_image(image, window_masks)
					
					text_image_score = self.calculate_text_anomaly_score(text_features, image_cls, normal=False)

					vision_anomaly_map = self.calculate_visual_anomaly_score(tokens, window_cls, window_masks)
					vision_anomaly_score = vision_anomaly_map.max()

					AC_score = 0.5 * (text_image_score + vision_anomaly_score)
					print(f"text: {text_image_score}")
					print(f"vision: {vision_anomaly_score}")
					return AC_score.detach().cpu().numpy()
			else:
				if shot == 0:
					#zero-shot AS apply multi-scale aggregation score across pixels
					window_masks_1 = self.gen_window_mask(kernel_size=32).squeeze().to(DEVICE)
					window_masks_2 = self.gen_window_mask(kernel_size=48).squeeze().to(DEVICE)
					window_masks = [window_masks_1, window_masks_2]
					window_cls, tokens, image_cls = self.encode_image(image, window_masks)
					
					window_cls_list = []
					for x in window_cls:
						[window_cls_list.append(window) for window in x]

					window_text_score = [self.calculate_text_anomaly_score(text_features, x, normal=True) for x in window_cls_list]
					image_text_score = self.calculate_text_anomaly_score(text_features, image_cls, normal=True)

					window_list = []
					scale_indices = []
					for x in window_masks:
						x = x.T #(length, num_mask) -> (num_mask, length)
						index = x.shape[0] if len(scale_indices) == 0 else scale_indices[-1] + x.shape[0]
						scale_indices.append(index)
						[window_list.append(window) for window in x]

					anomaly_score_map = []
					#distribute window_scale score to each patch
					cur_window_score = torch.zeros(15*15, device=DEVICE)
					cur_patch_weights = torch.zeros(15*15, device=DEVICE)
					for count, (score, window) in enumerate(zip(window_text_score, window_list)):
						#distribute score to a tensor
						#start a new scale then do average on previous scale
						if count in scale_indices:
							cur_window_score = cur_window_score / cur_patch_weights
							anomaly_score_map.append(cur_window_score)
							index += 1

							cur_window_score = torch.zeros(15*15, device=DEVICE)
							cur_patch_weights = torch.zeros(15*15, device=DEVICE)

						window = window.long() - 1
						temp_score = torch.zeros(15 * 15, device=DEVICE)
						temp_weight = torch.zeros(15 * 15, device=DEVICE)
						temp_score[window] = 1.0 / score
						temp_weight[window] = 1.0
						cur_window_score += temp_score
						cur_patch_weights += temp_weight
						count += 1

					#last scale
					cur_window_score = cur_window_score / cur_patch_weights
					anomaly_score_map.append(cur_window_score)

					#distribute image_scale score
					image_scale_score = torch.zeros(15*15, device=DEVICE)
					image_scale_score = torch.full((15*15,), 1.0 / image_text_score.item(), device=DEVICE)
					anomaly_score_map.append(image_scale_score)

					anomaly_score_map = torch.stack(anomaly_score_map, dim=0)
					anomaly_score_map = torch.mean(anomaly_score_map, dim=0)
					anomaly_score_map = 1.0 - 1.0 / anomaly_score_map

					anomaly_map = anomaly_score_map.reshape(15, 15).unsqueeze(0)
					anomaly_map = anomaly_map.unsqueeze(0)
					anomaly_map = F.interpolate(anomaly_map, size=(240, 240), mode='bilinear', align_corners=False)

					ret_map = anomaly_map.squeeze().detach().cpu().numpy()
					return ret_map
				else:
					#mess with reference images
					window_masks_1 = self.gen_window_mask(kernel_size=32).squeeze().to(DEVICE)
					window_masks_2 = self.gen_window_mask(kernel_size=48).squeeze().to(DEVICE)
					window_masks = [window_masks_1, window_masks_2]
					self.gen_reference_bank(ref_list, window_masks)

					window_cls, tokens, image_cls = self.encode_image(image, window_masks)
					
					window_cls_list = []
					for x in window_cls:
						[window_cls_list.append(window) for window in x]

					window_text_score = [self.calculate_text_anomaly_score(text_features, x, normal=True) for x in window_cls_list]
					image_text_score = self.calculate_text_anomaly_score(text_features, image_cls, normal=True)

					window_list = []
					scale_indices = []
					for x in window_masks:
						x = x.T #(length, num_mask) -> (num_mask, length)
						index = x.shape[0] if len(scale_indices) == 0 else scale_indices[-1] + x.shape[0]
						scale_indices.append(index)
						[window_list.append(window) for window in x]

					anomaly_score_map = []
					#distribute window_scale score to each patch
					cur_window_score = torch.zeros(15*15, device=DEVICE)
					cur_patch_weights = torch.zeros(15*15, device=DEVICE)
					for count, (score, window) in enumerate(zip(window_text_score, window_list)):
						#distribute score to a tensor
						#start a new scale then do average on previous scale
						if count in scale_indices:
							cur_window_score = cur_window_score / cur_patch_weights
							anomaly_score_map.append(cur_window_score)
							index += 1

							cur_window_score = torch.zeros(15*15, device=DEVICE)
							cur_patch_weights = torch.zeros(15*15, device=DEVICE)

						window = window.long() - 1
						temp_score = torch.zeros(15 * 15, device=DEVICE)
						temp_weight = torch.zeros(15 * 15, device=DEVICE)
						temp_score[window] = 1.0 / score
						temp_weight[window] = 1.0
						cur_window_score += temp_score
						cur_patch_weights += temp_weight
						count += 1

					#last scale
					cur_window_score = cur_window_score / cur_patch_weights
					anomaly_score_map.append(cur_window_score)

					#distribute image_scale score
					image_scale_score = torch.zeros(15*15, device=DEVICE)
					image_scale_score = torch.full((15*15,), 1.0 / image_text_score.item(), device=DEVICE)
					anomaly_score_map.append(image_scale_score)

					anomaly_score_map = torch.stack(anomaly_score_map, dim=0)
					anomaly_score_map = torch.mean(anomaly_score_map, dim=0)
					anomaly_score_map = 1.0 - 1.0 / anomaly_score_map

					visual_score_map = self.calculate_visual_anomaly_score(tokens, window_cls, window_masks)

					final_map = visual_score_map + anomaly_score_map
					final_map = final_map.reshape(15, 15).unsqueeze(0)
					final_map = final_map.unsqueeze(0)
					ret_map = F.interpolate(final_map, size=(240, 240), mode='bilinear', align_corners=False)

					return ret_map.squeeze().detach().cpu().numpy()


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

def run():
	model = WinCLIP(state_level, template_level).to(DEVICE)
	ds = ADDataset("bottle", DS_DIR, shot=1, preprocess=model.preprocess)
	ref_list, img, isAbno, indice, image_path = ds[3]
	ref_list = [x.unsqueeze(0) for x in ref_list]
	anomaly_map = model.forward("bottle", img.unsqueeze(0), shot=1, option="AS", ref_list=ref_list)

	# print(f"Truth: {isAbno}")
	# print(f"Predict score: {anomaly_score}")
	segment(anomaly_map, img)

def eval():
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
			mvtec_adds = ADDataset(object_name, DS_DIR, shot=0, preprocess=win_model.preprocess)
			dataloader = DataLoader(mvtec_adds, batch_size=1, num_workers=1, shuffle=False)
			for data in tqdm(dataloader, desc="Running: "):
				ref_list, img, isAbno, indice, img_path = data
				score = win_model.forward(object_name, img, ref_list, shot=0, option="AC")
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
    eval()