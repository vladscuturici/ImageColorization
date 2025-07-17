import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_score
from skimage.metrics import peak_signal_noise_ratio as psnr_score
import numpy as np

from model.model_architecture import ColorizationNet
from app.utils import preprocess_image, extract_l_channel, postprocess_output
from app.config import MODEL_PATH, DEVICE

def load_model(model_path):
    model = ColorizationNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.eval()
    return model

model = load_model(MODEL_PATH)


deeplab = deeplabv3_resnet50(weights="DEFAULT").eval().to(DEVICE)
faster_rcnn = fasterrcnn_resnet50_fpn(weights="DEFAULT").eval().to(DEVICE)

def extract_instance_feats(image_tensor):
    with torch.no_grad():
        detections = faster_rcnn([image_tensor])[0]
        if len(detections["boxes"]) == 0:
            return torch.zeros((1, 1, 256, 256)).to(DEVICE)

        masks = []
        for box in detections["boxes"]:
            mask = torch.zeros((256, 256)).to(DEVICE)
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 1.0
            masks.append(mask)

        instance_map = torch.stack(masks).sum(0, keepdim=True)
        instance_map = instance_map.unsqueeze(0)
        return instance_map


@torch.no_grad()
def predict(model, pil_image, style_image=None, gt_color=None, saturation_scale=1.4):
    # model = load_model(model_path)

    input_tensor = preprocess_image(pil_image).to(DEVICE)
    input_l = extract_l_channel(pil_image).to(DEVICE)

    seg_output = deeplab(input_tensor)['out']
    segmap = torch.argmax(seg_output, dim=1, keepdim=True).float()

    instance_feats = extract_instance_feats(input_tensor[0])

    style_feats = None
    if style_image:
        style_tensor = preprocess_image(style_image).to(DEVICE)
        with torch.no_grad():
            x1 = model.encoder["layer1"](style_tensor)
            x2 = model.encoder["layer2"](x1)
            x3 = model.encoder["layer3"](x2)
            x4 = model.encoder["layer4"](x3)
        style_feats = x4

    pred_ab, _ = model(input_tensor, segmap, instance_feats=instance_feats, style_feats=style_feats)
    pred_ab_upsampled = F.interpolate(pred_ab, size=input_l.shape[2:], mode='bilinear', align_corners=False)

    # colorized = postprocess_output(pred_ab_upsampled, input_l)

    pred_ab_scaled = torch.clamp(pred_ab_upsampled * saturation_scale, -1.0, 1.0)
    colorized = postprocess_output(pred_ab_scaled, input_l)

    gt_lab = gt_color.resize((256, 256)).convert("LAB")
    gt_lab_np = np.array(gt_lab).astype(np.float32)

    # L = gt_lab_np[..., 0:1] / 255.0
    # ab = (gt_lab_np[..., 1:] - 128) / 128.0

    gt_np = np.array(gt_color.resize((256, 256))).astype(np.float32)
    pred_np = np.array(colorized.resize((256, 256))).astype(np.float32)

    gt_arr = np.array(gt_color.resize((256, 256)).convert("RGB"))
    if np.std(gt_arr, axis=2).mean() < 1.0:
        psnr_val = None
        ssim_val = None
    else:
        gt_np = gt_arr.astype(np.float32)
        pred_np = np.array(colorized.resize((256, 256))).astype(np.float32)
        psnr_val = psnr_score(gt_np, pred_np, data_range=255)
        ssim_val = ssim_score(gt_np, pred_np, channel_axis=-1, data_range=255)

    return colorized, psnr_val, ssim_val
