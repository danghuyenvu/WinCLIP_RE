#constants parameters
import torch

MODEL_NAME = "ViT-B-16-plus-240"
PRETRAINED = "laion400m_e32"
RE_SIZE = 240
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
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
    "pipe_fryum"
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

DS_DIR = [
    "../Datasets/MVtecAD",
    "../Datasets/VisA"
    ]

EVAL_DES = [
    "../Results/MVtec-AD",
    "../Results/VisA"
]

datasets = {
    "mvtec-ad" : MVTEC_AD_OBJ,
    "visa" : VISA_OBJ
}