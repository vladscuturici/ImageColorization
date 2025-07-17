import torch
import numpy as np
import cv2
from PIL import Image
from torch import nn
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor
from app.config import DEVICE

def preprocess_image(pil_img):
    img = pil_img.convert("L").resize((256, 256))
    img_np = np.array(img)

    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor

def extract_l_channel(pil_img):
    l_tensor = torch.from_numpy(
        np.array(pil_img.convert("L").resize((256, 256)), dtype=np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)
    return l_tensor

def postprocess_output(pred_ab, input_l):
    pred_ab_np = pred_ab.squeeze().permute(1, 2, 0).cpu().numpy()
    ab = ((pred_ab_np * 128) + 128).astype(np.uint8)

    L = (input_l.squeeze().cpu().numpy() * 255).astype(np.uint8)[..., np.newaxis]

    lab = np.concatenate([L, ab], axis=2)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

def lab_to_rgb_tensor(L, ab):
    L = (L * 100).cpu().numpy()
    ab = (ab * 128).cpu().numpy()
    lab = np.concatenate([L, ab], axis=1)
    rgb = []

    for img in lab:
        img = np.transpose(img, (1, 2, 0)).astype(np.float32)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        rgb_img = np.clip(rgb_img / 255.0, 0, 1)
        rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        rgb.append(rgb_tensor)

    return torch.stack(rgb).to(DEVICE)
