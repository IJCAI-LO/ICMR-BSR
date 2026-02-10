from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from clip.adaptor import Adaptor


def gaussian_kernel(size, sigma=2.0):
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    y = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    x, y = torch.meshgrid(x, y, indexing='ij')

    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


from collections import OrderedDict
import torch
from torch import nn
from typing import List, Optional, Tuple, Union

# 你原来就有的
# LayerNorm, QuickGELU

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, return_attn: bool = False):
        """
        x: [L, B, C]
        return:
          - out:  [L, B, C]
          - attn: None or [B, heads, L, L]
        """
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)

        if not return_attn:
            out = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
            return out, None

        # need_weights=True 时返回注意力
        # PyTorch 不同版本对 average_attn_weights 支持不同，因此做兼容
        try:
            out, attn = self.attn(
                x, x, x,
                need_weights=True,
                attn_mask=attn_mask,
                average_attn_weights=False  # -> [B, heads, L, L]
            )
        except TypeError:
            # 老版本没有 average_attn_weights 参数 -> attn: [B, L, L] (平均过 heads)
            out, attn = self.attn(
                x, x, x,
                need_weights=True,
                attn_mask=attn_mask
            )
            # 统一成 [B, heads, L, L]：用 1 head 代替（至少能跑通）
            attn = attn.unsqueeze(1)

        # 某些版本返回 [B, L, L]，也统一成 [B,1,L,L]
        if attn.dim() == 3:
            attn = attn.unsqueeze(1)

        return out, attn

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        attn_out, attn = self.attention(self.ln_1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers

        # 原来你用 nn.Sequential，这里必须换成 ModuleList 才能取中间层 attn
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        print("resblocks: ", len(self.resblocks))

    def forward(
        self,
        x: torch.Tensor,
        feature_layers: Optional[List[int]] = None,
        visual_prompt: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        attn_layers: Optional[List[int]] = None,
        fearure_layers=None,  # 兼容你原来的拼写（外部若还在用）
    ):
        """
        x: [L,B,C]
        feature_layers: 1-based layer indices, e.g. [6,12,18,24]
        return_attn: whether to return attention maps
        attn_layers: 1-based layer indices to collect attn maps

        return:
          - if return_attn=False:
              - feature_layers is None -> x
              - else -> out(list[tensor])
          - if return_attn=True:
              - feature_layers is None -> (x, attn_out)
              - else -> (out, attn_out)
            where attn_out is list of [B, heads, L, L]
        """
        # backward compatibility for typo
        if feature_layers is None and fearure_layers is not None:
            feature_layers = fearure_layers

        out = []
        attn_out = []

        prefix_len = len(visual_prompt) if visual_prompt is not None else 0
        want_feats = feature_layers is not None
        want_attn = return_attn and (attn_layers is not None) and (len(attn_layers) > 0)

        feat_set = set(feature_layers) if want_feats else set()
        attn_set = set(attn_layers) if want_attn else set()

        for i in range(len(self.resblocks)):
            layer_id = i + 1  # 1-based

            # 你原来的 prompt 注入逻辑保留（一般 vision 不用，但不破坏其他地方）
            if i < prefix_len:
                # 注意：这里假设 visual_prompt 的形状和你原先一致
                x = torch.cat([visual_prompt[i:i + 1].repeat(x.size(0), 1, 1), x], dim=1)

            x, attn = self.resblocks[i](x, return_attn=(want_attn and layer_id in attn_set))

            if i < prefix_len:
                x = x[:, visual_prompt[i:i + 1].size(1):]

            if want_feats and (layer_id in feat_set):
                out.append(x)

            if want_attn and (layer_id in attn_set):
                attn_out.append(attn)  # [B, heads, L, L]

        if not return_attn:
            return x if feature_layers is None else out

        # return_attn=True
        if feature_layers is None:
            return x, attn_out
        return out, attn_out


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        print(self.positional_embedding.size())

    def forward(
            self,
            x: torch.Tensor,
            feature_layers: List[int] = [24],
            visual_prompt: Optional[torch.Tensor] = None,
            return_attn: bool = False,
            attn_layers: Optional[List[int]] = None
    ):
        """
        Default: return out_tokens_list
        If return_attn=True: return (out_tokens_list, attn_list)
          attn_list: list of [B, heads, L, L] collected at attn_layers

        IMPORTANT FIX:
          - Do NOT write back positional_embedding.data inside forward.
          - Use interpolated pos_embed as a local tensor.
        """
        # patchify
        x = self.conv1(x)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid**2]
        x = x.permute(0, 2, 1)  # [B, grid**2, width]

        # prepend CLS
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)  # [B, 1+N, width]

        # ---- positional embedding (NO in-place writeback) ----
        pos_embed = self.positional_embedding  # [1+N0, C] as parameter
        side = int((pos_embed.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        if side != new_side:
            # interpolate patch position embeddings
            # pos_embed[1:] -> [N0, C] -> [1, C, side, side] -> interp -> [N1, C]
            patch_pos = pos_embed[1:, :].reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0)  # [1,C,side,side]
            patch_pos = torch.nn.functional.interpolate(patch_pos, size=(new_side, new_side), mode="bilinear",
                                                        align_corners=False)
            patch_pos = patch_pos.squeeze(0).permute(1, 2, 0).reshape(new_side * new_side, -1)  # [N1,C]
            pos_embed = torch.cat([pos_embed[:1, :], patch_pos], dim=0)  # [1+N1, C]

        x = x + pos_embed.to(x.dtype)

        # optional extra visual prompt token (kept compatible, but doesn't touch Transformer prompt)
        if visual_prompt is not None:
            x = torch.cat([x, visual_prompt[:1].repeat(x.size(0), 1, 1)], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # [L,B,C]

        # ---- transformer forward ----
        if return_attn:
            # If you applied my Transformer patch: use forward_with_attn
            if hasattr(self.transformer, "forward_with_attn"):
                out, attns = self.transformer.forward_with_attn(
                    x,
                    feature_layers=feature_layers,
                    attn_layers=attn_layers,
                    visual_prompt=None
                )
            else:
                # fallback to your current signature (if you still use return_attn in Transformer.forward)
                out, attns = self.transformer(
                    x,
                    feature_layers=feature_layers,
                    visual_prompt=None,
                    return_attn=True,
                    attn_layers=attn_layers
                )
        else:
            out = self.transformer(x, feature_layers=feature_layers)
            attns = None

        # out: list of [L,B,C] -> [B,L,C]
        for i in range(len(out)):
            out[i] = out[i].permute(1, 0, 2)
            if visual_prompt is not None:
                out[i] = out[i][:, :-visual_prompt.size(1), :]

        if not return_attn:
            return out
        return out, attns


try:
    import scipy.ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# You must keep your original imports / definitions:
# ModifiedResNet, VisionTransformer, Transformer, LayerNorm,
# gaussian_kernel, Adaptor
# (Assuming they are in the same module scope as your original code.)


def anomaly_sfr(predict_map, sigma=3.0, thresh=0.5):
    if not _HAS_SCIPY:
        return predict_map

    B, C, H, W = predict_map.shape
    out = []

    for b in range(B):
        m = predict_map[b, 0]
        binary = (m > thresh).float()

        labeled, num = ndi.label(binary.detach().cpu().numpy())
        if num == 0:
            out.append(m.unsqueeze(0))
            continue

        smooth = torch.zeros_like(m)
        for i in range(1, num + 1):
            region_np = (labeled == i)
            dist_np = ndi.distance_transform_edt(region_np)

            region = torch.tensor(region_np, device=m.device)
            dist = torch.tensor(dist_np, device=m.device, dtype=m.dtype)

            smooth += region * torch.exp(-dist / sigma)

        smooth = smooth / (smooth.max() + 1e-6)
        out.append(smooth.unsqueeze(0))

    return torch.stack(out, dim=0)  # [B,1,H,W]





import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple, Union

import os
import re

def _safe_name(s: str, max_len: int = 120) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-\.]+", "_", s)
    return s[-max_len:]



class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int
    ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        # cache for inference-only attention (optional)
        self._cached_mid_attns = None

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    # ====== AF-CLIP insert ======
    def insert(self, args, tokenizer, device):
        # ===== Prompt Ensemble (domain-agnostic, works for industrial + medical + daily) =====
        # Normal (negative) templates
        use_simple_prompt =True  # True: without/with defect
        # False: multi-template photo prompts

        if use_simple_prompt:
            self.normal_templates = [
                "without defect."
            ]
            self.anomaly_templates = [
                "with defect."
            ]
        else:
            self.normal_templates = [
                "a photo of a normal object.",
                "a photo of an intact object.",
                "a photo without defects.",
                "a photo with no anomaly.",
                "a clean object."
            ]
            self.anomaly_templates = [
                "a photo of a defective object.",
                "a photo with defects.",
                "a photo with an anomaly.",
                "a damaged object.",
                "a photo with an irregularity."
            ]
        self.num_normal = len(self.normal_templates)
        self.num_anomaly = len(self.anomaly_templates)

        # tokenize all prompts (K = num_normal + num_anomaly)
        all_prompts = self.normal_templates + self.anomaly_templates
        self.state_prompt_tokens = tokenizer(all_prompts).to(device)

        self.tokenizer = tokenizer
        self.device = device
        self.prompt_len = args.prompt_len

        # learnable prompt prefix (kept trainable, same as your original)
        self.state_prompt_embedding = nn.Parameter(
            torch.empty(1, args.prompt_len, self.token_embedding.weight.shape[-1]).to(device)
        )
        nn.init.normal_(self.state_prompt_embedding, std=0.01)
        self.state_prompt_embedding.requires_grad_(True)

        # adaptor & other components (same as your original)
        self.adaptor = Adaptor(inplanes=self.visual.proj.shape[0], outplanes=self.visual.proj.shape[0]).to(device)
        self.memorybank = None
        self.memory_backbone = None
        self.gaussian_kernel = {
            '3': gaussian_kernel(size=3, sigma=4).to(device),
            '5': gaussian_kernel(size=5, sigma=4).to(device)
        }

    def _vis_init(self):
        """
        Initialize cache structures for stable id allocation.
        """
        if not hasattr(self, "_vis_cnt"):
            self._vis_cnt = 0
        if not hasattr(self, "_vis_id_map"):
            # bname -> id (int)
            self._vis_id_map = {}

    def _vis_alloc_id(self, bname: str, max_n: int):
        """
        Allocate a stable id for each bname (image).
        If bname already exists, return the same id.
        If total ids exceed max_n, return None (stop saving).
        """
        self._vis_init()

        if bname in self._vis_id_map:
            return self._vis_id_map[bname]

        if self._vis_cnt >= max_n:
            return None

        self._vis_cnt += 1
        self._vis_id_map[bname] = self._vis_cnt
        return self._vis_cnt

    # ----------------------------
    # Extract defect type for MPDD-style paths
    # ----------------------------
    def _infer_defect_from_path(self, p: str) -> str:
        """
        Infer defect/category name from file path.
        Works for MPDD-like:
          .../mpdd/<class>/test/<defect>/xxx.png
          .../mpdd/<class>/ground_truth/<defect>/xxx_mask.png

        If not found, returns "unknown".
        """
        if p is None:
            return "unknown"
        p = str(p).replace("\\", "/")

        # Heuristic: the directory right after 'test' or 'ground_truth'
        # e.g. .../ground_truth/hole/000_mask.png -> hole
        m = re.search(r"/ground_truth/([^/]+)/", p)
        if m:
            return _safe_name(m.group(1))

        m = re.search(r"/test/([^/]+)/", p)
        if m:
            return _safe_name(m.group(1))

        # fallback: if you have other known tokens
        for k in ["hole", "scratches", "crack", "broken", "oil", "stain"]:
            if f"/{k}/" in p:
                return k

        return "unknown"

    @torch.no_grad()
    def _maybe_save_stage(self, image, pixel_map, stage: str, args, img_path=None, extra: dict = None):
        """
        Save map as .npy for later high-quality plotting.

        Output naming (stable per image):
          <id4>_<bname>_image.npy
          <id4>_<bname>_<stage>_map.npy
          <id4>_<bname>_<stage>_<extra_key>.npy

        IMPORTANT: id4 is allocated ONCE per bname, not per stage.

        Parameters:
          image:      torch.Tensor [B,3,H,W] or None
          pixel_map:  torch.Tensor [B,1,h,w] in [0,1] ideally
          img_path:   str or list[str], used to name and infer defect
          extra:      dict of extra numpy/tensor maps (e.g. corr_seed, gateM)
        """
        if int(getattr(args, "save_vis", 0)) != 1:
            return

        self._vis_init()
        max_n = int(getattr(args, "save_vis_max", 50))

        save_dir = getattr(args, "save_vis_dir", "./vis_cache")
        os.makedirs(save_dir, exist_ok=True)

        # pick first sample in batch -> stable bname
        bname = "sample"
        raw_path = None
        if img_path is not None:
            if isinstance(img_path, (list, tuple)):
                raw_path = img_path[0]
                bname = _safe_name(os.path.basename(raw_path))
            else:
                raw_path = img_path
                bname = _safe_name(os.path.basename(raw_path))

        # ADD defect to avoid MPDD name collision: hole__000.png, scratches__000.png
        defect = self._infer_defect_from_path(raw_path)
        bname = f"{defect}__{bname}"

        # allocate stable id for this bname
        idx = self._vis_alloc_id(bname, max_n)
        if idx is None:
            return

        # stable prefix for THIS IMAGE
        prefix_img = f"{idx:04d}_{bname}"
        self._last_vis_prefix = prefix_img  # e.g. "0001_hole__000.png"

        prefix_stage = f"{idx:04d}_{bname}_{stage}"

        # (1) save map
        pm = pixel_map[0, 0].detach().float().cpu().numpy()
        np.save(os.path.join(save_dir, f"{prefix_stage}_map.npy"), pm)

        # (2) save image once (avoid rewriting repeatedly)
        if int(getattr(args, "save_vis_image", 1)) == 1 and (image is not None):
            img_path_out = os.path.join(save_dir, f"{prefix_img}_image.npy")
            if not os.path.exists(img_path_out):
                im = image[0].detach().float().cpu().numpy()  # [3,H,W]
                np.save(img_path_out, im)

        # (3) save extras for this stage
        if extra is not None:
            for k, v in extra.items():
                if torch.is_tensor(v):
                    v = v.detach().float().cpu().numpy()
                np.save(os.path.join(save_dir, f"{prefix_stage}_{k}.npy"), v)

    @torch.no_grad()
    def _save_corr_seed_map(self, corr: torch.Tensor, pixel_map: torch.Tensor, args, img_path=None):
        """
        corr: [B,N,N] correlation matrix (no CLS)
        pixel_map: [B,1,H,W] current map (used to choose seed)

        Save:
          <id>_<bname>_rcs_corr_seed.npy
        """
        if getattr(args, "save_vis", 0) != 1 or getattr(args, "save_corr_seed", 0) != 1:
            return

        if corr is None or (not torch.is_tensor(corr)):
            return

        B, _, H, W = pixel_map.shape
        P = pixel_map[:, 0].reshape(B, -1)  # [B,N]
        seed_idx = torch.argmax(P, dim=1)  # [B]

        b0 = 0
        sidx = int(seed_idx[b0].item())
        corr_row = corr[b0, sidx]  # [N]
        corr_map = corr_row.reshape(H, W)

        # normalize to [0,1] for visualization
        cm = corr_map.detach().float()
        cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-8)

        # IMPORTANT: image=None to avoid saving dummy/overwriting
        self._maybe_save_stage(
            image=None,
            pixel_map=pixel_map,
            stage="rcs_only",
            args=args,
            img_path=img_path,
            extra={"corr_seed": cm.cpu().numpy()}
        )

    # ====== text ======
    def encode_state_prompt(self):
        """
        Return:
          text_features: [K, embed_dim], where
            first num_normal are normal templates,
            last  num_anomaly are anomaly templates.
        """
        # token embeddings: [K, 77, C]
        state_x = self.token_embedding(self.state_prompt_tokens).type(self.dtype)

        # prepend learnable prompt (same for all K prompts)
        state_x = torch.cat(
            [self.state_prompt_embedding.repeat(state_x.size(0), 1, 1), state_x],
            dim=1
        )[:, :77, :]

        state_x = state_x + self.positional_embedding.type(self.dtype)
        state_x = state_x.permute(1, 0, 2)  # [L, K, C]
        state_x = self.transformer(state_x)
        state_x = state_x.permute(1, 0, 2)  # [K, L, C]
        state_x = self.ln_final(state_x).type(self.dtype)

        # EOT index for each prompt (offset by prompt_len because we prepended learnable tokens)
        idx = self.prompt_len + self.state_prompt_tokens.argmax(dim=-1)  # [K]
        state_x = state_x[torch.arange(state_x.shape[0]), idx] @ self.text_projection  # [K, embed_dim]
        return state_x

    def get_trainable_parameters(self):
        return list([self.state_prompt_embedding]) + list(self.adaptor.parameters())

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    # ====== image ======
    def encode_image(self, image, feature_layers=None, return_attn: bool = False, attn_layers=None):
        # ViT supports return_attn in your modified VisionTransformer
        if isinstance(self.visual, VisionTransformer):
            return self.visual(
                image.type(self.dtype),
                feature_layers,
                visual_prompt=None,
                return_attn=return_attn,
                attn_layers=attn_layers
            )
        else:
            return self.visual(image.type(self.dtype), feature_layers)

    # ===== neighbor aggregation =====
    def aggerate_neighbor(self, x, patchsize, stride=1):
        if patchsize == 1:
            return x
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]
        padding = patchsize // 2
        b, l, c = x.size()
        h = w = int(math.sqrt(l))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        x = torch.nn.functional.unfold(x, kernel_size=patchsize, padding=padding, stride=stride)
        x = x.permute(0, 2, 1).reshape(-1, c, patchsize * patchsize).permute(0, 2, 1)
        kernel = self.gaussian_kernel[str(patchsize)].reshape(1, patchsize * patchsize, 1)
        x = torch.sum(x * kernel, dim=1).reshape(b, l, c)
        x = torch.cat([cls_token, x], dim=1)
        return x

    def aggerate_neighbors(self, img_tokens):
        img_token_list = []
        for img_token in img_tokens:
            for r in [1, 3, 5]:
                new_img_token = self.aggerate_neighbor(img_token, r)
                img_token_list.append(new_img_token)
        return img_token_list

    # =========================
    # (A) Training-safe encoder: only tokens (NO attention returned)
    # =========================
    def detect_encode_image(self, image, args):
        base_tokens = self.encode_image(image, args.feature_layers)  # list of [B,1+N,C]
        score_tokens = self.aggerate_neighbors(base_tokens)
        score_tokens = [self.visual.ln_post(self.adaptor(t)) @ self.visual.proj for t in score_tokens]
        return score_tokens  # list of [B,1+N,embed]

    # =========================
    # (B) Inference-only attention fetcher (separate, won't affect training)
    # =========================
    @torch.no_grad()
    def get_mid_attns(self, image, args):
        if not isinstance(self.visual, VisionTransformer):
            return []
        _, mid_attns = self.encode_image(
            image,
            feature_layers=args.feature_layers,
            return_attn=True,
            attn_layers=getattr(args, "attn_rcs_layers", self.default_attn_rcs_layers)
        )
        # mid_attns: list of [B,heads,L,L]
        return mid_attns

    # =========================
    # Gated Attention-RCS (recommended for anomaly detection)
    # =========================
    def build_corr_from_attn(self, attn_list: List[torch.Tensor], tau: float = 1.0):
        """
        attn_list: list of [B, heads, L, L]
        return corr: [B, N, N] (exclude CLS)
        """
        corr_sum, cnt = None, 0
        for attn in attn_list:
            # [B,heads,L,L] -> [B,L,L]
            a = attn.mean(dim=1)
            a = a[:, 1:, 1:]  # [B,N,N]
            if tau != 1.0:
                a = (a / tau).softmax(dim=-1)
            corr_sum = a if corr_sum is None else (corr_sum + a)
            cnt += 1
        return corr_sum / max(cnt, 1)

    def attn_rcs_refine_predict_map_af(
            self,
            predict_map: torch.Tensor,
            attn_list: List[torch.Tensor],
            lambda_rcs: float = 0.2,
            seed_ratio: float = 0.1,
            tau: float = 1.0,
            add_identity: bool = True,
            bg_quantile: float = 0.7,  # 背景抑制强度
            preserve_energy: bool = True,  # 守恒总能量（关键！）
            calib: str = "zscore"  # "zscore" or "minmax" or "none"
    ):
        """
        AF-RCS:
          1) top-k anomaly seeds only
          2) propagate with attn corr
          3) suppress background via quantile clamp
          4) preserve energy to keep calibration stable (helps AP/F1/CLS)
          5) per-image calibration
        """
        B, _, H, W = predict_map.shape
        P = predict_map.view(B, -1)  # [B,N]

        corr = self.build_corr_from_attn(attn_list, tau=tau)  # [B,N,N]
        if add_identity:
            N = corr.size(-1)
            I = torch.eye(N, device=corr.device, dtype=corr.dtype).unsqueeze(0)
            corr = corr + I
            corr = corr / corr.sum(dim=-1, keepdim=True)

        # seeds
        k = max(1, int(P.size(1) * seed_ratio))
        thresh = torch.topk(P, k, dim=1).values[:, -1].unsqueeze(1)
        P_seed = P * (P >= thresh).float()

        # propagate seeds
        P_prop = torch.bmm(corr, P_seed.unsqueeze(-1)).squeeze(-1)  # [B,N]

        # background clamp (抑制正常区域被抬高)
        q = torch.quantile(P, bg_quantile, dim=1, keepdim=True)  # [B,1]
        P_prop = torch.clamp(P_prop - q, min=0.0)

        # energy preserving (让分布不漂移，AP/F1/CLS会稳很多)
        if preserve_energy:
            src = P_seed.sum(dim=1, keepdim=True) + 1e-6
            dst = P_prop.sum(dim=1, keepdim=True) + 1e-6
            P_prop = P_prop * (src / dst)

        # residual fusion
        P_new = (1 - lambda_rcs) * P + lambda_rcs * P_prop

        # per-image calibration (score 标定，提升 AP/F1/CLS)
        if calib == "zscore":
            mu = P_new.mean(dim=1, keepdim=True)
            std = P_new.std(dim=1, keepdim=True) + 1e-6
            P_new = (P_new - mu) / std
            P_new = torch.sigmoid(P_new)  # map back to [0,1]
        elif calib == "minmax":
            mn = P_new.min(dim=1, keepdim=True).values
            mx = P_new.max(dim=1, keepdim=True).values
            P_new = (P_new - mn) / (mx - mn + 1e-6)

        return P_new.view(B, 1, H, W)

    # =========================
    # AF-CLIP segmentation forward (TRAIN-SAFE!)
    # returns img_tokens as 3rd output for patch_alignment_loss
    # =========================
    def detect_forward_seg(self, image, args):
        """
        Always return 3 values:
          cls_label, predict_map, img_tokens

        Additionally cache:
          - self._mid_attns_cache: for Attention-RCS
          - self._pix_logits_cache: for bg calibration
        """
        # ---- text ----
        text_features = self.encode_state_prompt()
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        # ---- image tokens + (optional) mid attentions ----
        mid_attns = None
        use_attn = (not self.training) and (getattr(args, "use_attn_rcs", 1) == 1) and isinstance(self.visual,
                                                                                                  VisionTransformer)

        if use_attn:
            attn_layers = getattr(args, "attn_rcs_layers", [6, 7, 8, 9])
            try:
                base_tokens, mid_attns = self.encode_image(
                    image,
                    feature_layers=args.feature_layers,
                    return_attn=True,
                    attn_layers=attn_layers
                )
            except Exception:
                base_tokens = self.encode_image(image, args.feature_layers)
                mid_attns = None
        else:
            base_tokens = self.encode_image(image, args.feature_layers)

        # cache for detect_forward
        self._mid_attns_cache = mid_attns

        # ---- AF-CLIP neighbor aggregation + adaptor ----
        img_tokens = self.aggerate_neighbors(base_tokens)
        img_tokens = [self.visual.ln_post(self.adaptor(t)) @ self.visual.proj for t in img_tokens]

        # ---- compute scores ----
        scores = 0
        for img_token in img_tokens:
            img_token = torch.nn.functional.normalize(img_token, dim=-1)
            scores += torch.matmul(img_token, text_features.t()) / 0.07
        scores = scores / len(img_tokens)

        # ---- prompt ensemble mean ----
        logits_normal = scores[..., :self.num_normal].mean(dim=-1)  # [B,1+N]
        logits_anom = scores[..., self.num_normal:].mean(dim=-1)  # [B,1+N]

        # cache pixel logits for bg calibration
        self._pix_logits_cache = torch.stack([logits_normal, logits_anom], dim=-1)  # [B,1+N,2]

        prob = torch.softmax(torch.stack([logits_normal, logits_anom], dim=-1), dim=-1)
        cls_label = prob[:, 0, 1].view(-1)  # [B]
        predict_map = prob[:, 1:, 1]  # [B,N]

        b, l = predict_map.size()
        h = w = int(math.sqrt(l))
        predict_map = predict_map.reshape(b, 1, h, w)

        return cls_label, predict_map, img_tokens
    # =========================
    # memorybank (unchanged)
    # =========================
    def store_memory(self, image, args):
        img_tokens = self.encode_image(image, args.memory_layers)
        img_tokens = self.aggerate_neighbors(img_tokens)
        b, l, c = img_tokens[0].size()

        new_bank = [
            torch.nn.functional.normalize(t[:, 1:], dim=-1).reshape(-1, c)
            for t in img_tokens
        ]

        if self.memorybank is None:
            self.memorybank = new_bank
        else:
            self.memorybank = [
                torch.cat([old, nb], dim=0) for old, nb in zip(self.memorybank, new_bank)
            ]

    def detect_forward_memorybank(self, image, args):
        scores = 0
        img_tokens = self.encode_image(image, args.memory_layers)
        img_tokens = self.aggerate_neighbors(img_tokens)
        for i, img_token in enumerate(img_tokens):
            img_token = torch.nn.functional.normalize(img_token, dim=-1)
            score = (1 - torch.matmul(img_token, self.memorybank[i].T)).min(dim=-1)[0] / 2
            scores += score[:, 1:]
        scores = scores / len(img_tokens)
        cls_label = torch.max(scores, dim=-1)[0]
        b, l = scores.size()
        h = w = int(math.sqrt(l))
        predict_map = scores.reshape(b, 1, h, w)
        print(f"[detect_forward_memorybank] scores mean/std={scores.mean().item():.6f}/{scores.std().item():.6f}, "
              f"cls_mb mean={cls_label.mean().item():.6f}")

        return cls_label, predict_map

    # =========================
    # Inference forward: fusion first, then Attention-RCS (gated)
    # 2026/1/31 9：44  now
    # =========================
    def detect_forward(self, image, args, img_path=None):
        cls_label, pixel_map, _ = self.detect_forward_seg(image, args)
        cls_label = cls_label.clamp(0.0, 1.0)

        # ===== SAVE baseline =====
        self._maybe_save_stage(image, pixel_map, stage="baseline", args=args, img_path=img_path)

        if self.training:
            return cls_label, pixel_map

        # -----------------------------------------------------
        # (0) Background Calibration (optional) -> base_map
        # -----------------------------------------------------
        base_map = pixel_map
        if getattr(args, "use_bg_calib", 1) == 1:
            # (你的 bgcalib 原代码保持不变，只是最后写到 base_map)
            beta0 = float(getattr(args, "bg_beta", 0.52))
            bg_q = float(getattr(args, "bg_quantile", 0.75))
            base_q = float(getattr(args, "bg_base_quantile", 0.95))

            logits2 = getattr(self, "_pix_logits_cache", None)
            if logits2 is not None and torch.is_tensor(logits2):
                logits_normal = logits2[..., 0]
                logits_anom = logits2[..., 1]
                la = logits_anom[:, 1:]

                p = base_map[:, 0].reshape(base_map.size(0), -1)

                pmax = p.max(dim=1, keepdim=True)[0]
                pmean = p.mean(dim=1, keepdim=True)
                compact = (pmax / (pmean + 1e-6)).clamp(1.0, 50.0)
                beta = (beta0 * (1.2 - 0.3 * ((compact - 3.0) / 7.0).clamp(0.0, 1.0))).clamp(0.45, 0.70)

                thr = torch.quantile(p, q=bg_q, dim=1, keepdim=True)
                bg_mask = (p <= thr)

                bases = []
                for b in range(la.size(0)):
                    vals = la[b][bg_mask[b]]
                    base = la[b].mean() if vals.numel() < 16 else torch.quantile(vals, q=base_q)
                    bases.append(base)
                base = torch.stack(bases, dim=0).view(-1, 1).clamp(min=0.0)

                la_cal = la - beta * base
                logits_anom_cal = torch.cat([logits_anom[:, :1], la_cal], dim=1)
                logits2_cal = torch.stack([logits_normal, logits_anom_cal], dim=-1)

                prob = torch.softmax(logits2_cal, dim=-1)
                cls_label = prob[:, 0, 1].clamp(0.0, 1.0)

                pm = prob[:, 1:, 1]
                bsz, L = pm.size()
                h = w = int(math.sqrt(L))
                base_map = pm.reshape(bsz, 1, h, w)

                self._maybe_save_stage(image, base_map, stage="bgcalib", args=args, img_path=img_path)

        # -----------------------------------------------------
        # helper: SRR/SFR (与你主控一致：sigma/thresh)
        # -----------------------------------------------------
        def apply_sfr_sigma_thresh(x_map):
            """
            SRR with quantile anchor + SOFT gate.
            - still uses per-image quantile to avoid "no trigger"
            - gate is continuous in [0,1], not binary
            """
            if getattr(args, "use_sfr", 1) != 1:
                return x_map, None, None  # y, gate_soft, gate_hard

            sigma = float(getattr(args, "sfr_sigma", 3.0))

            # q=0.98 => use top 2% as "anchor threshold"
            gate_q = float(getattr(args, "srr_gate_q", 0.98))

            # temperature controls softness (bigger => smoother, smaller => closer to hard)
            gate_temp = float(getattr(args, "srr_gate_temp", 0.08))

            x = x_map.clamp(0.0, 1.0)  # [B,1,H,W]
            B, _, H, W = x.shape
            p = x[:, 0].reshape(B, -1)  # [B,HW]

            thr = torch.quantile(p, q=gate_q, dim=1, keepdim=True)  # [B,1]
            thr = thr.view(B, 1, 1, 1)

            # ---- SOFT gate: sigmoid around thr ----
            # when x==thr => gate=0.5
            # gate_temp smaller => sharper
            gate_soft = torch.sigmoid((x - thr) / (gate_temp + 1e-12))

            # optional: hard gate for debug / ablation
            gate_hard = (x >= thr).float()

            # gaussian kernel
            ksz = int(max(3, (6 * sigma + 1)))
            if ksz % 2 == 0:
                ksz += 1
            half = ksz // 2

            coords = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
            g1 = torch.exp(-(coords ** 2) / (2 * (sigma ** 2) + 1e-12))
            g1 = g1 / (g1.sum() + 1e-12)
            g2 = (g1[:, None] * g1[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,ksz,ksz]

            x_pad = torch.nn.functional.pad(x, (half, half, half, half), mode="reflect")
            x_blur = torch.nn.functional.conv2d(x_pad, g2)

            # ---- SRR mixing by SOFT gate ----
            y = gate_soft * x_blur + (1.0 - gate_soft) * x

            if int(getattr(args, "debug_srr", 0)) == 1:
                print(
                    "[DEBUG][SRR]",
                    "q=", gate_q,
                    "thr mean=", float(thr.mean().item()),
                    "gate_soft mean=", float(gate_soft.mean().item()),
                    "gate_hard mean=", float(gate_hard.mean().item()),
                    "x max=", float(x.max().item())
                )

            return y, gate_soft, gate_hard

        # -----------------------------------------------------
        # (A) ONLY-RCS
        # -----------------------------------------------------
        rcs_only_map = base_map
        corr = None
        mid_attns = getattr(self, "_mid_attns_cache", None)
        if getattr(args, "use_attn_rcs", 1) == 1 and (mid_attns is not None) and len(mid_attns) > 0:
            lam = float(getattr(args, "lambda_attn_rcs", 1.5))
            tau = float(getattr(args, "attn_rcs_tau", 1.0))

            topk = int(getattr(args, "rcs_topk", 128))
            gamma = float(getattr(args, "rcs_gamma", 1.2))
            add_I = int(getattr(args, "rcs_add_I", 1)) == 1

            B, _, H, W = base_map.shape
            P = base_map.view(B, -1)

            corr_sum, cnt = None, 0
            for attn in mid_attns:
                a = attn.mean(dim=1)
                a = a[:, 1:, 1:]
                if tau != 1.0:
                    a = (a / tau).softmax(dim=-1)
                corr_sum = a if corr_sum is None else (corr_sum + a)
                cnt += 1
            corr = corr_sum / max(cnt, 1)

            N = corr.size(-1)
            k = min(max(topk, 1), N)
            vals, idx = torch.topk(corr, k=k, dim=-1)
            sparse = torch.zeros_like(corr)
            sparse.scatter_(-1, idx, vals)
            corr = sparse

            if add_I:
                I = torch.eye(N, device=corr.device, dtype=corr.dtype).unsqueeze(0)
                corr = corr + I

            if gamma != 1.0:
                corr = torch.clamp(corr, min=0.0) ** gamma

            corr = corr / (corr.sum(dim=-1, keepdim=True) + 1e-6)

            Pref = torch.bmm(corr, P.unsqueeze(-1)).squeeze(-1).view(B, 1, H, W)

            # residual form (lam can be >1)
            rcs_only_map = base_map + lam * (Pref - base_map)

            self._maybe_save_stage(image, rcs_only_map, stage="rcs_only", args=args, img_path=img_path)

            # corr-seed evidence (optional)
            if getattr(args, "save_vis", 0) == 1 and getattr(args, "save_corr_seed", 0) == 1:
                Pnow = rcs_only_map[:, 0].reshape(B, -1)
                seed_idx = torch.argmax(Pnow, dim=1)
                b0 = 0
                sidx = int(seed_idx[b0].item())
                corr_row = corr[b0, sidx].reshape(H, W)
                cm = corr_row.detach().float()
                cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-8)
                self._maybe_save_stage(
                    image=None, pixel_map=rcs_only_map, stage="rcs_only",
                    args=args, img_path=img_path, extra={"corr_seed": cm.cpu().numpy()}
                )

        # -----------------------------------------------------
        # (B) ONLY-SRR (apply SRR directly on base_map)
        # -----------------------------------------------------
        srr_only_map, gate_only_soft, gate_only_hard = apply_sfr_sigma_thresh(base_map)

        if getattr(args, "use_sfr", 1) == 1:
            self._maybe_save_stage(
                image, srr_only_map, stage="srr_only", args=args, img_path=img_path,
                extra={
                    "gateM": gate_only_soft[0, 0].detach().float().cpu().numpy(),  # 画机制图用这个
                    # 你想保留 hard gate 也可以：
                    "gateM_hard": gate_only_hard[0, 0].detach().float().cpu().numpy()
                } if gate_only_soft is not None else None
            )

        # -----------------------------------------------------
        # (C) FULL = RCS + SRR (apply SRR on rcs_only_map)
        # -----------------------------------------------------
        full_map, gate_full_soft, gate_full_hard = apply_sfr_sigma_thresh(rcs_only_map)

        if getattr(args, "use_sfr", 1) == 1:
            self._maybe_save_stage(
                image, full_map, stage="full", args=args, img_path=img_path,
                extra={
                    "gateM": gate_full_soft[0, 0].detach().float().cpu().numpy(),
                    "gateM_hard": gate_full_hard[0, 0].detach().float().cpu().numpy()
                } if gate_full_soft is not None else None
            )
        else:
            # if no SRR, full equals rcs_only
            full_map = rcs_only_map
            self._maybe_save_stage(image, full_map, stage="full", args=args, img_path=img_path)

        # -----------------------------------------------------
        # return: by default return FULL output
        # -----------------------------------------------------
        pixel_map = full_map

        # memorybank fusion (optional) - usually should be applied AFTER you decide what "final" is
        if self.memorybank is not None:
            cls_mb, pixel_mb = self.detect_forward_memorybank(image, args)
            w = float(getattr(args, "alpha", 0.2))
            w = max(0.0, min(1.0, w))
            pixel_map = (1 - w) * pixel_map + w * pixel_mb
            cls_label = ((1 - w) * cls_label + w * cls_mb).clamp(0.0, 1.0)

        return cls_label, pixel_map

    # ===== standard CLIP forward =====
    def forward(self, image, text):
        image_features = self.encode_image(image)
        if isinstance(image_features, (list, tuple)):
            image_features = image_features[0]
        text_features = self.encode_text(text)
        if isinstance(text_features, (list, tuple)):
            text_features = text_features[0]

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()