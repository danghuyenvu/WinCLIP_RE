import open_clip
import torch

MODEL_NAME = "ViT-B-16-plus-240"
PRETRAINED = "laion400m_e32"

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# Save weights to disk
torch.save(model.state_dict(), "winCLIP_ViT_B16P_laion400m.pt")