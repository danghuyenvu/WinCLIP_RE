from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob
import os
from PIL import Image
import torch
import csv
from params import MVTEC_AD_OBJ, MEAN, STD, VISA_OBJ

class MVtecADDataset(Dataset):
	"""
	Dataset class for MVtec AD data set
	"""
	def __init__(self, obj_type, data_dir, spatial_size=240, mode="train", shot=0, preprocess=None):
		super(MVtecADDataset, self).__init__()
		self.shot = shot
		self.object_type = obj_type if obj_type != "all" else MVTEC_AD_OBJ
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
		if isinstance(self.object_type, list):
			#get all images path of all objects
			for x in self.object_type:
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
		gt = None
		if os.path.basename(folder_path) == "good":
			isAbno = np.array([0], dtype=np.float32)
			gt = torch.zeros((240, 240))
		else:
			#get groundtruth masks
			gt_path = image_path.replace("test", "ground_truth")
			base, ext = os.path.splitext(gt_path)
			gt_path = base + "_mask" + ext
			gt = Image.open(gt_path)
			gt = gt.resize((240,240), resample=Image.NEAREST)
			to_tensor = transforms.ToTensor()
			gt = to_tensor(gt)
			
		img = self.transform_image(image_path, self.preprocess) if self.preprocess is not None else self.transform_image(image_path, self.pre_transform)

		return ref_list, img, isAbno, indice, gt

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
	

class VisADataset(Dataset):
	"""
	Dataset class for VisA data set
	"""
	def __init__(self, obj_type, data_dir, spatial_size=240, mode="train", shot=0, preprocess=None):
		super(VisADataset, self).__init__()
		self.shot = shot
		self.object_type = obj_type if obj_type != "all" else VISA_OBJ
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
		self.img_label = []
		self.img_mask = []
		if isinstance(self.object_type, list):
			#get all images path, label, masks of all objects
			for x in self.object_type:
				cur_obj = os.path.join(self.data_dir, x)
				anno_dir = os.path.join(cur_obj, "image_anno.csv")
				with open(anno_dir) as csvfile:
					reader = csv.DictReader(csvfile)
					for row in reader:
						self.img_path.append(os.path.join(self.data_dir, row["image"]))
						self.img_label.append(np.array([0], dtype=np.float32) if row["label"] == 'normal' else np.array([1], dtype=np.float32))
						self.img_mask.append(os.path.join(self.data_dir, row["mask"]) if row["label"] != 'normal' else None)
			return

		type_dir = os.path.join(self.data_dir, self.object_type)
		anno_dir = os.path.join(type_dir, "image_anno.csv")
		with open(anno_dir) as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				self.img_path.append(os.path.join(self.data_dir, row["image"]))
				self.img_label.append(np.array([0], dtype=np.float32) if row["label"] == 'normal' else np.array([1], dtype=np.float32))
				self.img_mask.append(None if row["label"] == 'normal' else os.path.join(self.data_dir, row["mask"]))

	def convert_rgb(self, x):
		return x.convert('RGB')
	
	def __getitem__(self, indice):
		ref_list = []
		img = None
		gt = None
		if self.shot != 0:
			#get the first shot number of images for reference
			normal_image = self.img_path[:self.shots]

			if isinstance(self.object_type, list):
				raise ValueError("Only all for Zero shot")
			
			for x in normal_image:
				if self.preprocess is not None:
					ref_list.append(self.transform_image(x, self.preprocess))
				else:
					ref_list.append(self.transform_image(x, self.pre_transform))

		if indice + self.shot >= len(self.img_path):
			raise ValueError("Invalid indice")
		
		image_path = self.img_path[indice + self.shot]
		image_label = self.img_label[indice + self.shot]
		gt_path = self.img_mask[indice + self.shot]

		if gt_path is not None:
			gt = Image.open(gt_path)
			to_tensor = transforms.ToTensor()
			gt = to_tensor(gt)
		
		image = self.transform_image(image_path, self.preprocess) if self.preprocess is not None else self.transform_image(image_path, self.pre_transform)

		return ref_list, image, image_label, indice, gt

	def __len__(self):
		return len(self.img_path) - self.shot

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