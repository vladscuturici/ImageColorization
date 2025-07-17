import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet50
from einops import rearrange
import torch

class PixelDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(in_channels, 512, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(512 + 1024, 256, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(256 + 512, 128, 4, 2, 1)
        self.out = nn.Conv2d(128, 2, 3, padding=1)

    def forward(self, x4, x3, x2):
        x = self.up4(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        out = torch.tanh(self.out(x))
        return out

class ColorQueryDecoder(nn.Module):
    def __init__(self, feature_dim, num_queries=64):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, feature_dim))
        self.transformer = nn.Transformer(d_model=feature_dim, batch_first=True, dropout=0.1)
        self.pos_encoding = nn.Parameter(torch.randn(64, 1, feature_dim))
        self.query_norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        B, C, H, W = features.shape
        x = features.flatten(2).permute(0, 2, 1)
        x = x + self.pos_encoding[:x.size(0)]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        color_features = self.transformer(queries, x)
        color_features = self.query_norm(color_features)
        color_features = F.dropout(color_features, p=0.1, training=self.training)

        return color_features


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.norm = nn.BatchNorm2d(norm_nc, affine=False)
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, 3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, 3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, 3, padding=1)

    def forward(self, x, segmap):
        segmap = segmap.to(x.device)
        normalized = self.norm(x)
        segmap = nn.functional.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out

class InstanceFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, global_feats, instance_feats):
        combined = torch.cat([global_feats, instance_feats], dim=1)
        return self.conv(combined)

class ColorizationNet(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        resnet = resnet50(pretrained=True)
        resnet = resnet.to(self.device)
        self.encoder = nn.ModuleDict({
            "layer1": nn.Sequential(*list(resnet.children())[:5]),
            "layer2": resnet.layer2,
            "layer3": resnet.layer3,
            "layer4": resnet.layer4,
        })

        self.encoder.to(self.device)

        self.pixel_decoder = PixelDecoder(2048).to(self.device)
        self.color_query_decoder = ColorQueryDecoder(2048).to(self.device)
        self.spade = SPADE(2048, 1).to(self.device)
        self.instance_fusion = InstanceFusion(2048).to(self.device)
        self.guidance_proj = nn.Linear(2048, 2048).to(self.device)
        self.instance_gate = nn.Conv2d(2048, 2048, kernel_size=1)
        self.query_gate = nn.Conv2d(2048, 2048, kernel_size=1).to(self.device)

    def forward(self, x_gray, segmap, instance_feats=None, style_feats=None):
        x1 = self.encoder["layer1"](x_gray)
        x2 = self.encoder["layer2"](x1)
        x3 = self.encoder["layer3"](x2)
        x4 = self.encoder["layer4"](x3)

        x4 = self.spade(x4, segmap)

        if instance_feats is not None:
            if instance_feats.shape[1] == 1:
                instance_feats = instance_feats.repeat(1, x4.shape[1], 1, 1)
            if instance_feats.shape[2:] != x4.shape[2:]:
                instance_feats = F.interpolate(instance_feats, size=x4.shape[2:], mode='bilinear', align_corners=False)
            gate = torch.sigmoid(self.instance_gate(x4))
            x4 = self.instance_fusion(x4, instance_feats * gate)

        color_query = self.color_query_decoder(x4)

        query_map = self.guidance_proj(color_query)
        query_map = query_map.permute(0, 2, 1).contiguous()
        query_map = query_map.view(x4.size(0), x4.size(1), x4.size(2), x4.size(3))

        query_map = query_map / (query_map.norm(dim=1, keepdim=True) + 1e-6)
        query_map = F.dropout(query_map, p=0.3, training=self.training)

        gate = torch.sigmoid(self.query_gate(x4))
        x4 = x4 + 0.02 * gate * query_map

        ab_channels = self.pixel_decoder(x4, x3, x2)
        return ab_channels, color_query