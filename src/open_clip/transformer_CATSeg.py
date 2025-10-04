from collections import OrderedDict
import math
from dataclasses import asdict
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.models.group_vit import GroupingPrompt, GroupingBlockSet
from .reins import Reins, reinscfg
from .utils import to_2tuple


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(
            x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps
        )
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape,
                         self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        scaled_cosine=False,
        scale_heads=False,
        logit_scale_max=math.log(1.0 / 0.01),
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(
            torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1)))
            )
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, need_weights=False, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight,
                           self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(
                F.normalize(q, dim=-1), F.normalize(k,
                                                    dim=-1).transpose(-1, -2)
            )
            logit_scale = torch.clamp(
                self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        n_head: int = 8,
        n_queries: int = 256,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(
            d_model, n_head, kdim=context_dim, vdim=context_dim
        )
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        is_cross_attention: bool = False,
        attn_mask: torch.Tensor = None,
        attn_layer: str = "multi_head_attention",
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.attn_mask = attn_mask

    def attention(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]
        # return self.attn(v_x, v_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(
                self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(
                self, "ln_1_kv") and v_x is not None else None
        )

        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x,
                           v_x=v_x, attn_mask=attn_mask)
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.feature_block = None
        self.width = width
        self.layers = layers
        self.heads = heads
        self.grad_checkpointing = True

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(layers)
            ]
        )

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        save_feature=False,
        feature_list=[2, 5, 8, 11],
    ):
        if save_feature:
            self.feature_block = []
        for idx, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
            if save_feature and idx in feature_list:
                self.feature_block.append(x.permute(1, 0, 2))
        return x


class LearnablePrompt(nn.Module):
    def __init__(self, cnt, length, width):
        assert cnt == len(length)
        super(LearnablePrompt, self).__init__()
        self.p = nn.ParameterList()
        for idx, l in enumerate(length):
            prompt_shape = (l, width)
            self.p.append(nn.Parameter(torch.randn(prompt_shape)))
            nn.init.normal_(self.p[idx], mean=0, std=0.25)

    def forward(self, x, grid_size=(24, 24), idx=0):
        cat_prompt = self.p[idx].unsqueeze(0).permute(
            1, 0, 2).repeat(1, x.shape[1], 1) # shape = [length, B, width] = [16, 32, 768]
        x = torch.cat(
            [x[: (grid_size[0] * grid_size[1] + 1), :, :], cat_prompt], dim=0) 
        # shape = [grid ** 2 + 1 + length, B, width] = [197 + 16, 32, 768] = [213, 32, 768]
        return x


class VisionPromptTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        ls_init_value: float = None,
        global_average_pool: bool = False,
        attentional_pool: bool = False,
        n_queries: int = 256,
        attn_pooler_heads: int = 8,
        output_dim: int = 512,  # check
        patch_dropout: float = 0.0,
        input_patchnorm: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,
        # prompt
        prompt_length: list = None,
        prompt_layer_idx: list = None,
        feature_layer: list = [6, 10, 14, 22],
        # group
        group_layer_idx: list = None,
        group_length: list = None,
        # pseudo
        pseudo_layer: list = None,
        image_scale: int = 2,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height,
                          image_width // patch_width)
        self.output_dim = output_dim
        self.width = width
        self.num_heads = heads
        self.layers = layers
        print("grid", image_height, patch_height, self.grid_size)
        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = input_patchnorm

        if input_patchnorm:
            patch_input_dim = patch_height * patch_width * 3
            self.patchnorm_pre_ln = LayerNorm(patch_input_dim)
            self.conv1 = nn.Linear(patch_input_dim, width)
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )

        # class embeddings and positional embeddings
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0]
                                * self.grid_size[1] + 1, width)
        ) # shape = [patch_num, width]
        self.register_buffer('positional_embedding_backup', self.positional_embedding.detach().clone())

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = (
            PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()
        )

        #####################################

        #####################################

        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.prompt_length = prompt_length
        self.prompt_layer_idx = prompt_layer_idx
        self.feature_layer = feature_layer
        self.group_layer_idx = group_layer_idx
        self.group_length = group_length

        self.pseudo_layer = pseudo_layer
        self.mix_noise = 1
        self.noise_std = [0.25]

        self.noise_prob = [1]

        if prompt_layer_idx is not None:
            self.learnable_prompt = LearnablePrompt(
                len(prompt_layer_idx), prompt_length, width
            )

        # assert len(group_layer_idx) == len(group_length)
        if group_layer_idx is not None:
            self.group_prompt = GroupingPrompt(
                cnt=len(group_layer_idx), length=group_length, width=self.width
            )
            self.group_block_set = GroupingBlockSet(
                dim=self.width,
                out_dim=self.width,
                num_group_tokens=group_length,
                num_output_groups=group_length,
            )

        self.global_average_pool = global_average_pool
        self.reins = Reins(**asdict(reinscfg))
        width = self.width

        if patch_size == 14 or patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.GroupNorm(1, width),
                nn.ConvTranspose2d(width, width,
                                   kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.GroupNorm(1, width),
                nn.Conv2d(width, width,
                          kernel_size=3, stride=1, padding=1),
            )

            self.fpn3 = nn.Sequential(
                nn.GroupNorm(1, width),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.GroupNorm(1, width),
                nn.MaxPool2d(kernel_size=4, stride=4),
            )

        if attentional_pool:
            self.attn_pool = AttentionalPooler(
                output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries
            )
            self.ln_post = norm_layer(output_dim)
            self.proj = nn.Parameter(scale * torch.randn(output_dim, output_dim))
        else:
            self.attn_pool = None
            self.ln_post = norm_layer(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("if self.global_average_pool:", self.global_average_pool)
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def contrastive_loss(self, features, labels, temperature=0.5):
        # n, c, h, w = features.shape

        # 1. reshape feature
        # features = features.view(n, c, -1).permute(0, 2, 1)  # shape: [n, h*w, c]
        features_normalized = F.normalize(features, dim=2)

        # 2. calc similarity
        similarity_matrix = (
            torch.bmm(features_normalized, features_normalized.transpose(1, 2))
            / temperature
        )

        # 3. make mask
        # labels = labels.view(n, -1)  # shape: [n, h*w]
        mask = (labels.unsqueeze(1) == labels.unsqueeze(2)).float()

        # 4. get loss
        loss = (-similarity_matrix * mask + (1 - mask)
                * similarity_matrix.exp()).mean()

        return loss

    def forward_features(self, x: torch.Tensor, x_seg: torch.Tensor, train = False, new_grid_size = (24, 24)):
        # ViT-B-16-224
            # x.shape = [grid ** 2 + 1, bs, width]
            # x_seg.shape = [grid ** 2, bs, width]
            # new_grid_size = (14, 14)
        patch_features = []
        self.res["inner_fpn_feat"] = []
        batch_size, patch_width, grid_height, grid_width = self.res["shape"] # 32, 768, 14, 14
        for i, layer in enumerate(self.transformer.resblocks.children()):
            if self.prompt_layer_idx is not None and i in self.prompt_layer_idx: # 8
                x = self.learnable_prompt(
                    x, grid_size=new_grid_size, idx=self.prompt_layer_idx.index(i)
                ) # shape = [grid ** 2 + 1 + length, B, width] = [213, 32, 768]
                x = layer(x) # shape = [grid ** 2 + 1 + length, B, width] = [213, 32, 768]

            else:
                x = layer(x) # shape = [grid ** 2 + 1, B, width] = [197, 32, 768]
            
            if i == self.layers - 2: # origin -2 feature
                
                patch_features = [x.permute(1, 0, 2)[
                    :, : (new_grid_size[0] * new_grid_size[1] + 1), :
                ] for _ in range(4)] # shape = [4, B, grid_num + 1, width] = [4, 32, 197, 768]


                self.res["tokens"] = x.permute(1, 0, 2)[
                    :, 1: (new_grid_size[0] * new_grid_size[1] + 1), :
                ]  # [B, grid_num, width] = [32, 196, 768]

        # (48, 48), (24, 24), (12, 12), (6, 6)
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for idx, patch_feature in enumerate(patch_features):
            # print(x.shape)
            z = patch_feature[:, 1:new_grid_size[0] * new_grid_size[1] + 1].permute(0, 2, 1)
            z = z.view(batch_size, self.width, new_grid_size[0], new_grid_size[1])
            self.res["inner_fpn_feat"].append(
                ops[idx](z))

        # x.shape = [grid ** 2 + 1 + prompt_length, batch_size, width]
        self.res["x"] = x.permute(1, 0, 2)  # LND -> NLD
        self.res["x_seg"] = x_seg.permute(1, 0, 2)  # LND -> NLD
        self.res["feat"] = patch_features

    def forward_head(self, train):
        x = self.res["x"]
        x = self.ln_post(x[:, 0, :])
        device = x.device
        if self.proj is not None:
            x = x @ self.proj
        self.res["per_logits"] = x

        if self.prompt_layer_idx is not None and train:
            labels = torch.zeros((x.shape[0], self.patch_num**2)).to(device)

            self.res["loss"] = torch.tensor(0).float().to(device)
            for i in range(len(self.res["seg_feat"])):
                self.res["loss"] += self.contrastive_loss(
                    self.res["seg_feat"][i], labels, temperature=0.5
                )

    def inject_text(text_features: torch.Tensor):
        self.text_features = text_features  # [B, 2, embed_size]

    def _get_updated_positional_embedding(self, x, new_side, side):
        """
        根据输入 x 的形状动态调整 positional_embedding
        """
        if side != new_side:
            # 动态调整 positional_embedding
            new_pos = (
                self.positional_embedding[1:, :]
                .reshape(-1, side, side, x.shape[-1])
                .permute(0, 3, 1, 2)
            )
            new_pos = F.interpolate(new_pos, (new_side, new_side), mode="bilinear")
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            new_pos = torch.cat([self.positional_embedding[:1, :], new_pos[0]], 0)
        else:
            # 如果输入形状与 positional_embedding 匹配，直接使用
            new_pos = self.positional_embedding

        return new_pos

    def forward(
        self, x: torch.Tensor, mask=None, proj=False, save_feature=False, train=False
    ):
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1

        self.res = dict()
        batch_size, image_channel, _, _ = x.shape

        if self.input_patchnorm: # False
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(
                x.shape[0],
                x.shape[1],
                self.grid_size[0],
                self.patch_size[0],
                self.grid_size[1],
                self.patch_size[1],
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
            x = self.patchnorm_pre_ln(x)

        else:
            x = self.conv1(x)  # shape = [B, width, grid_height, grid_width] = [32, 768, 14, 14]
            _, _, grid_height, grid_width = x.shape # 14, 14
            self.res["shape"] = x.shape # shape = [B, width, grid_height, grid_width] = [32, 768, 14, 14]
            x = x.reshape(x.shape[0], x.shape[1], -1) # shape = [B, width, grid ** 2] = [32, 768, 196]
            x = x.permute(0, 2, 1)  # shape = [B, grid ** 2, width] = [32, 196, 768]
        # class embeddings and positional embeddings
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype) # shape = [B, 1, width] = [32, 1, 768]
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        # shape = [B, grid ** 2 + 1, width] = [32, 197, 768]

        side = int((self.positional_embedding.shape[0] - 1) ** 0.5) # grid = 14
        new_side = int((x.shape[1] - 1) ** 0.5) # grid = 14
        positional_embedding = self._get_updated_positional_embedding(x, new_side, side) # shape = [grid ** 2 + 1, width] = [197, 768]

        x = x + positional_embedding.to(x.dtype) # shape = [B, grid ** 2 + 1, width] = [32, 197, 768]
        ##################################
        tokens_list = []
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        # NLD -> LND 
        x = x.permute(1, 0, 2) # shape = [grid ** 2 + 1, B, width] = [197, 32, 768]

        # x = self.transformer(x, save_feature=save_feature)
        # Prompt Tuning
        # forward_features
        x_seg = x[1:, :, :] # shape = [grid ** 2, B, width] = [196, 32, 768]
        self.forward_features(
            x, x_seg, train, new_grid_size=(new_side, new_side))
        # self.forward_head(train)

        # forward_head
        x = self.res["x"] # shape = [B, grid_num + prompt_length + 1, width] = [32, 213, 768]
        seg_group = self.res["x_seg"] # shape = [B, grid_num, width] = [32, 196, 768]
        # x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x[:, 0, :])
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x) 
            # pooled shape = [B, 1, width] = [32, 1, 768]
            # tokens shape = [B, grid_num + prompt_length, width] = [32, 212, 768]
            pooled = self.ln_post(pooled)

        output_layers = []
        output_token_layers = []
        if self.res.get("feat") is not None:
            feat_list = self.res.get("feat")
            for i in feat_list:
                _pooled, _tokens = self._global_pool(i)
                output_layers.append(self.ln_post(_pooled)) # shape = [4, B, 1, width] = [4, 32, 1, 768]
                output_token_layers.append(self.ln_post(_tokens)) # shape = [4, B, grid_num, width] = [4, 32, 196, 768]
            output_seg_layers = output_token_layers

        if self.proj is not None:
            seg_group = seg_group @ self.proj # shape = [B, grid_num, output_dim] = [32, 196, 512]
            pooled = pooled @ self.proj # shape = [B, 1, output_dim] = [32, 1, 512]
            if 'tokens' in self.res and self.res["tokens"] is not None:
                tokens = self.res["tokens"] # shape = [B, grid_num, width] = [32, 196, 768]
                # tokens = tokens @ self.proj
            for i in range(output_layers.__len__()):
                output_layers[i] = output_layers[i] @ self.proj # shape = [batch_size, 1, output_dim] = [32, 1, 512]
                output_seg_layers[i] = output_seg_layers[i] @ self.proj # shape = [batch_size, grid_num, output_dim] = [32, 196, 512]
        self.res["tokens_list"] = tokens_list if tokens_list is not None else None
        self.res["pooled"] = pooled # shape = [B, 1, output_dim] = [32, 1, 512]
        self.res["seg_group"] = seg_group # shape = [B, grid_num, output_dim] = [32, 196, 512]
        self.res["tokens"] = tokens if tokens is not None else None # shape = [B, grid_num, width] = [32, 196, 768]
        self.res["output_layers"] = output_layers # shape = list of [B, 1, output_dim] = [4, 32, 1, 512]
        self.res["output_token_layers"] = output_token_layers # list of [B, grid_num, width] = [4, 32, 196, 768]
        self.res["seg_feat"] = output_seg_layers # list of [B, grid_num, output_dim] = [4, 32, 196, 512]
        self.res["fpn_feat"] = []

        for fpn_feature in self.res["inner_fpn_feat"]:
            fpn_feature = fpn_feature.permute(0, 2, 3, 1).view(
                fpn_feature.size(0), -1, fpn_feature.size(1))
            # [B, H*W, embed_size] -> [B, H*W, output_size]
            fpn_feature = fpn_feature @ self.proj
            self.res["fpn_feat"].append(fpn_feature)
        self.res['inner_fpn_feat'] = None

        if self.output_tokens:
            return self.res  # pooled, tokens

        if mask and not proj and save_feature:
            return self.res  # tokens_list, pooled, tokens
        if mask and not proj:
            return self.res  # tokens_list, pooled, tokens, output_layers
        if mask and proj:
            return self.res  # tokens_list, pooled, tokens @ self.proj, output_layers
        return self.res  # pooled, output_layers


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        ls_init_value: float = None,
        global_average_pool: bool = False,
        attentional_pool: bool = False,
        n_queries: int = 256,
        attn_pooler_heads: int = 8,
        output_dim: int = 512,
        patch_dropout: float = 0.0,
        input_patchnorm: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height,
                          image_width // patch_width)
        self.output_dim = output_dim
        print("grid", image_height, patch_height, self.grid_size)
        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = input_patchnorm

        if input_patchnorm:
            patch_input_dim = patch_height * patch_width * 3
            self.patchnorm_pre_ln = LayerNorm(patch_input_dim)
            self.conv1 = nn.Linear(patch_input_dim, width)
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )

        # class embeddings and positional embeddings
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0]
                                * self.grid_size[1] + 1, width)
        )
        # TODO: why not they don't use sincos_pos?
        # pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
        # self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = (
            PatchDropout(
                patch_dropout) if patch_dropout > 0.0 else nn.Identity()
        )

        #####################################

        #####################################

        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.global_average_pool = global_average_pool
        if attentional_pool:
            self.attn_pool = AttentionalPooler(
                output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries
            )
            self.ln_post = norm_layer(output_dim)
            self.proj = nn.Parameter(
                scale * torch.randn(output_dim, output_dim))
        else:
            self.attn_pool = None
            self.ln_post = norm_layer(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("if self.global_average_pool:", self.global_average_pool)
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def forward(self, x: torch.Tensor, mask=None, proj=False, save_feature=False):
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        b, c, _, _ = x.shape
        if self.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(
                x.shape[0],
                x.shape[1],
                self.grid_size[0],
                self.patch_size[0],
                self.grid_size[1],
                self.patch_size[1],
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.grid_size[0]
                          * self.grid_size[1], -1)
            x = self.patchnorm_pre_ln(x)
            x = self.conv1(x)
        else:
            # print("x",x.shape, self.input_patchnorm)
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            # print("x",x.shape, self.input_patchnorm)
            # shape = [*, width, grid ** 2]
            x = x.reshape(x.shape[0], x.shape[1], -1)
            # print("x",x.shape, self.input_patchnorm)
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # print("x_permute", x.shape)

        # class embeddings and positional embeddings
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        ##################################
        tokens_list = []
        # penultimate = []
        for mask_scale in mask:
            x_select = []
            mask_scale = mask_scale.T
            mask_num, l = mask_scale.shape
            class_index = torch.zeros((mask_scale.shape[0], 1), dtype=torch.int32).to(
                mask_scale
            )
            mask_scale = torch.cat((class_index, mask_scale.int()), dim=1)

            for i in mask_scale:
                x_select.append(torch.index_select(x, 1, i.int()))
            x_scale = torch.cat(x_select)  #

            x_scale = self.patch_dropout(x_scale)
            x_scale = self.ln_pre(x_scale)
            # print("x_scale", x_scale.shape)
            x_scale = x_scale.permute(1, 0, 2)  # NLD -> LND
            x_scale = self.transformer(x_scale)
            x_scale = x_scale.permute(1, 0, 2)  # LND -> NLD
            # print(x_scale.shape)
            if self.attn_pool is not None:
                x_scale = self.attn_pool(x_scale)
                x_scale = self.ln_post(x_scale)
                pooled, tokens = self._global_pool(x_scale)
            else:
                pooled, tokens = self._global_pool(x_scale)
                pooled = self.ln_post(pooled)

            if self.proj is not None:
                pooled = pooled @ self.proj

            if self.output_tokens:
                return pooled, tokens
            tokens_list.append(pooled.reshape(
                (mask_num, b, 1, -1)).permute(1, 0, 2, 3))

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x, save_feature=save_feature)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        if mask and not proj and save_feature:
            return tokens_list, pooled, tokens, self.transformer.feature_block
        if mask and not proj:
            return tokens_list, pooled, tokens
        if mask and proj:
            return tokens_list, pooled, tokens @ self.proj
        return pooled


##########################################################################################################################
class TextPromptTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        ls_init_value: float = None,
        output_dim: int = 512,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        embed_cls: bool = False,
        pad_id: int = 0,
        output_tokens: bool = False,
        feature_layer: list = [5],
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.feature_layer = feature_layer

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        self.register_buffer(
            "attn_mask", self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)
        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=self.transformer.width**-0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(
            cls_mask.shape, dtype=cast_dtype, device=cls_mask.device
        )
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(
            cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = (
                attn_mask[None, :seq_len, :seq_len] +
                cls_mask[:, :seq_len, :seq_len]
            )

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        x = self.transformer(
            x, attn_mask=attn_mask, save_feature=True, feature_list=self.feature_layer
        )
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(
                x.shape[0]), text.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens
        print("textpeeeeeeeeexxxxxxxxxxxxxxxxxxxeeeeeeed",
              pooled.shape, tokens.shape)
        return pooled, self.transformer.feature_block


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        ls_init_value: float = None,
        output_dim: int = 512,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        embed_cls: bool = False,
        pad_id: int = 0,
        output_tokens: bool = False,
        feature_layer: list = [5],
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.feature_layer = feature_layer

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        # self.transformer = PromptTransformer(
        #
        # )
        self.ln_final = norm_layer(width)

        self.register_buffer(
            "attn_mask", self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)
        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=self.transformer.width**-0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(
            cls_mask.shape, dtype=cast_dtype, device=cls_mask.device
        )
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(
            cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = (
                attn_mask[None, :seq_len, :seq_len] +
                cls_mask[:, :seq_len, :seq_len]
            )

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        print('text enter transformer')
        x = self.transformer(x, attn_mask=attn_mask)
        print('text out transformer')
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(
                x.shape[0]), text.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens
        print("textpeeeeeeeeexxxxxxxxxxxxxxxxxxxeeeeeeed",
              pooled.shape, tokens.shape)
        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        context_length: int = 77,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_dim: int = 512,
    ):

        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    is_cross_attention=True,
                )
                for _ in range(layers)
            ]
        )

        self.register_buffer(
            "attn_mask", self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, image_embs, text_embs):
        text_embs = text_embs.permute(1, 0, 2)  # NLD -> LNDsq
        image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
        seq_len = text_embs.shape[0]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(
                    resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len]
                )
                text_embs = checkpoint(
                    cross_attn, text_embs, image_embs, image_embs, None
                )
            else:
                text_embs = resblock(
                    text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len]
                )
                text_embs = cross_attn(
                    text_embs, k_x=image_embs, v_x=image_embs)

        x = text_embs.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
