# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import math

from typing import List, Tuple, Type


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4,
                               transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)
                 ).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,  # 下采样
    ) -> None:
        """
        A transformer decoder 尝试对一个输入图片使用带有位置embedding的查询
        由多个transformer block组成, 每个block包含两个attention模块.
        输入是图像的embedding、图像的position embedding和 点的embedding,
        输出是处理后的点的embedding和处理后的图像的embedding。
        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    # 在第一个循环中 i=0， 说明在TwoWayAttentionBlock前向传播过程中第一次进self attn
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,  # 传入的是token = output_tokens + prompt_tokens
    ) -> Tuple[Tensor, Tensor]:
        """
        前向传播过程:
        (1) 将图像的embedding和position embedding 分别经过一个线性层，
            得到image_embedding 和 image_pe。
        (2) 将点嵌入的embedding经过一个线性层,得到point_embedding。
        (3) 对 image_embedding 和 point_embedding 进行 transformer block处理,
            得到经过处理的 image_embedding 和 point_embedding。
        (4) 对经过处理的 image_embedding 和 point_embedding 进行交叉注意力，
            得到经过处理的 point_embedding 和 image_embedding。

        Args:
            image_embedding (torch.Tensor): 图像嵌入张量，形状为 B x embedding_dim x h x w。
            image_pe (torch.Tensor): 图像的位置编码张量，与 image_embedding 具有相同的形状。
            point_embedding (torch.Tensor): 查询点的嵌入张量，形状为 B x N_points x embedding_dim。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 经过处理的 point_embedding 和 image_embedding。

        """
        # Flatten image embedding to B x N_image_tokens x C
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        # image embedding 对应的 position embedding
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,  # 第一次添加时, queries与query_pe相同
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        # # 最后一个cross attn Final attention layer from the points to the image
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    #  TwoWayAttentionBlock = LayerNorm + Multi-Head Attention + MLP
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: 
        (1) self-attention of sparse inputs,
        (2) cross attention of sparse inputs to dense inputs,
        (3) mlp block on sparse inputs, 
        (4) cross attention of dense inputs to sparse
        inputs.
        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:

        # 第一个Self attention 模块。
        # 第一轮本身queries==query_pe
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # 第一个 Cross attention block。 tokens attending to image embedding
        # q, k, v不再是来源于同一个序列,而是多个序列. queries + query_pe充当q, k与v都由 keys提供
        # tokens to image embedding意味着，将token作为q, image_embedding 作为 k与v
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # 第二个 Cross attention block。 image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    一个允许下采样embedding size的attention层
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        # B x N_heads x N_tokens x C_per_head  C_per_head表示一个head中有多少个channel
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class WrapperMaskDecoder(torch.nn.Module):
    def __init__(self, img_size=336, prompt_embed_dim=256,  embed_size=768, patch_size=14, n_queries=2):
        super(WrapperMaskDecoder, self).__init__()
        self.prompt_embed_dim = prompt_embed_dim
        self.side = int(img_size // patch_size)
        self.grid_size = self.side * self.side
        self.emb_proj = torch.nn.Linear(embed_size, prompt_embed_dim)
        self.pe_proj = torch.nn.Linear(embed_size, prompt_embed_dim)
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        # self.proj = nn.Linear(embed_size, prompt_embed_dim)
        # self.n_queries = n_queries
        # self.sparse_prompt_query = nn.Parameter(1, self.n_queries, embed_size)
        # self.t_n_emb, self.t_a_emb = build_text_kv(prompt_templates, model)

    def forward(self, img_embed, img_pe, sparse_prompt_embeddings, dense_prompt_embeddings=None):
        # if sparse_prompt_embeddings is None:
        #     sparse_prompt_embeddings = self.sparse_prompt_query @ self.proj

        device = img_embed.device

        if dense_prompt_embeddings is None:
            dense_prompt_embeddings = torch.empty(1, self.prompt_embed_dim, self.side, self.side).to(device)

        img_embed = self.emb_proj(img_embed).permute(0, 3, 1, 2)
        img_pe = self.pe_proj(img_pe).permute(0, 2, 1)
        mask = self.mask_decoder(img_embed,
                                 img_pe,
                                 sparse_prompt_embeddings,
                                 dense_prompt_embeddings,
                                 False,)
        return mask


if __name__ == "__main__":
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    r_path = '/chencanyu-shcdt/huggingface_hub/sam_vit_l_0b3195.pth'
    prompt_embed_dim = 256
    wrapper_mask_decoder = WrapperMaskDecoder()
    with open(r_path, "rb") as f:
        checkpoint = torch.load(f)
        mask_decoder_state_dict = {k.replace(
            'mask_decoder.', ''): v for k, v in checkpoint.items() if 'mask_decoder' in k}
        wrapper_mask_decoder.mask_decoder.load_state_dict(
            mask_decoder_state_dict, strict=False)

    img_size = 336
    embed_size = 768
    patch_size = 14

    side = int(img_size // patch_size)
    grid_size = side * side

    bs = 32
    img_pe = torch.zeros(bs, grid_size, embed_size)
    img_embed = torch.randn(bs, side, side, embed_size)

    # sparse_prompt_embeddings = torch.empty(1, 0, prompt_embed_dim)
    sparse_prompt_embeddings = torch.randn(1, 32, prompt_embed_dim)
    # print(f"sparse_prompt_embeddings = {sparse_prompt_embeddings.shape}")
    dense_prompt_embeddings = torch.empty(1, prompt_embed_dim, side, side)

    result = wrapper_mask_decoder(
        img_embed, img_pe, sparse_prompt_embeddings, dense_prompt_embeddings)

    print(result[0].shape)

    # mask_decoder = MaskDecoder(
    #     num_multimask_outputs=3,
    #     transformer=TwoWayTransformer(
    #         depth=2,
    #         embedding_dim=prompt_embed_dim,
    #         mlp_dim=2048,
    #         num_heads=8,
    #     ),
    #     transformer_dim=prompt_embed_dim,
    #     iou_head_depth=3,
    #     iou_head_hidden_dim=256,
    # )

    # grid_size = 576
    # embed_size = 256
    # patch_size = 14

    # side = int(grid_size // patch_size)

    # img_pe = torch.zeros(1, embed_size, grid_size, )
    # img_embed = torch.randn(1, embed_size, side, side)
    # # sparse_prompt_embeddings = torch.randn(1, 10, prompt_embed_dim)
    # sparse_prompt_embeddings = torch.empty(1, 0, prompt_embed_dim)
    # dense_prompt_embeddings = torch.zeros(1, prompt_embed_dim, side, side)

    # result = mask_decoder(img_embed,
    #                       img_pe,
    #                       sparse_prompt_embeddings,
    #                       dense_prompt_embeddings,
    #                       False,)

    # print(result[0].shape)
