from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob
import os
from PIL import Image

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
DS_NAME = "mvtec-ad"

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