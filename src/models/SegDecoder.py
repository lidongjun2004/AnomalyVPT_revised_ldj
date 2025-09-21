from torch import nn
import torch
from src.open_clip.transformer import AttentionalPooler
from src.eva_clip import create_model_and_transforms
from src.models.sam import WrapperMaskDecoder
import torch.nn.functional as F
import math


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, sigma=0.5):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, tensor):
        # input: [Bs, 2, H, W]
        # 创建高斯核
        channels = tensor.shape[1]  # 获取通道数
        device = tensor.device  # 获取输入张量的设备

        x = torch.arange(self.kernel_size, dtype=torch.float32, device=device) - self.kernel_size // 2
        x = torch.exp(-(x ** 2) / (2 * self.sigma ** 2))  # 1D 高斯分布
        x = x / x.sum()  # 归一化
        gaussian_kernel = x[:, None] @ x[None, :]  # 2D 高斯核
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)  # 扩展为卷积核
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)  # 扩展到多通道

        # 使用卷积实现高斯模糊
        return F.conv2d(tensor, gaussian_kernel, padding=self.kernel_size // 2, groups=channels)

class GELUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., 
                norm_layer=nn.LayerNorm, subln=False
            ):
        super(GELUFFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.linear2(self.act(self.linear1(x))))
        return x

class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., 
                norm_layer=nn.LayerNorm, subln=False
            ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.upsample(x)


class ConvDecoder(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvDecoder, self).__init__()
        # 假设ViT的输出特征图的通道数为768
        self.decoder1 = DecoderBlock(768, 256)
        self.upsample1 = UpsampleBlock(scale_factor=2)
        self.decoder2 = DecoderBlock(256, 128)
        self.upsample2 = UpsampleBlock(scale_factor=2)
        self.decoder3 = DecoderBlock(128, num_classes)
        self.upsample3 = UpsampleBlock(scale_factor=3.5)

    def forward(self, x):
        # x: [32, 768, 24, 24]
        out = self.decoder1(x)
        out = self.upsample1(out)
        out = self.decoder2(out)
        out = self.upsample2(out)
        out = self.decoder3(out)
        out = self.upsample3(out)
        return out


class GroupDecoder(nn.Module):
    def __init__(self, dim=768, num_classes=2):
        super(GroupDecoder).__init__()
        self.dim = dim
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, t, scale=1 / 0.07):
        """

                Args:
                    x: shape [B, L, C]

                Returns:

                """
        # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)  # [B, num_classes]
        logits = x @ t.t() * scale
        return logits


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=1, d_model=768, nhead=12, dim_feedforward=4, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, t):
        for layer in self.layers:
            residual = x
            x = layer(x, t)
            x = residual + x
            x = self.norm(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.01, proj_drop=0.01):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B_q, N_q, C = x_q.shape
        B_kv, N_kv, C = x_kv.shape

        q = self.q(x_q).reshape(B_q, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x_kv).reshape(B_kv, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerBlock, self).__init__()
        self.attn_type = 'self'  # 'self' | 'cross'
        if self.attn_type == 'self':
            self.self_attn = nn.MultiheadAttention(d_model, nhead)
        elif self.attn_type == 'cross':
            self.self_attn = CrossAttention(dim=d_model, num_heads=nhead)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.gelu = QuickGELU()
        # self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ln_pre = nn.LayerNorm(d_model)
        self.ln_post = nn.LayerNorm(d_model)
        # self.swiglu = SwiGLU(in_features=d_model, hidden_features=8)
        self.swiglu = SwiGLU(in_features=d_model, hidden_features=int(d_model * 4.))
        # self.initialize()

    def forward(self, x, t):
        # Multi-head self-attention
        if self.attn_type == 'cross':
            attn_output, _ = self.self_attn(x, t)
        else:
            attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.ln_pre(x)

        # Feed-forward network
        # ff_output = self.linear2(self.dropout(self.gelu(self.linear1(x))))
        ff_output = self.swiglu(x)
        x = x + self.dropout(ff_output)
        x = self.ln_post(x)

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, 0.25)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)


class FFN(nn.Module):
    def __init__(self, embed_size):
        super(FFN, self).__init__()
        self.mlp_ratio = 4.0
        self.fc1 = nn.Linear(embed_size, embed_size * self.mlp_ratio) # 8
        self.gelu = QuickGELU()
        self.fc2 = nn.Linear(embed_size * self.mlp_ratio, embed_size) # 8
        self.initialize()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, 0.25)
                nn.init.constant_(m.bias, 0)

@torch.no_grad
def build_text_kv(prompt_templates, model):
     # use prompt_templates to initial kv
        # prompt_templates be like [['a crop photo of perfect xxx', 'a small photo of good xxx', ...], as t_n
        #                           ['a crop photo of defect xxx', 'a small photo of bad xxx', ...]], as t_a
        # use model as text encoder to build text embedding
        # ensemble to fixed size as n_heads

    # return 
    #   t_n_emb: [n_lens, embed_size]
    #   t_a_emb: [a_lens, embed_size]

    t_n_emb = torch.empty(0)
    t_a_emb = torch.empty(0)
    
    from src.open_clip import get_tokenizer
    # from src.eva_clip import get_tokenizer
    tokenizer = get_tokenizer('ViT-L-14-336')
    # tokenizer = get_tokenizer('EVA02-CLIP-L-14-336')

    for prompt in prompt_templates[0]:
        tokens = tokenizer(prompt)
        t_n_emb = torch.concat([t_n_emb, model.encode_text(tokens)])
    
    for prompt in prompt_templates[1]:
        tokens = tokenizer(prompt)
        t_a_emb = torch.concat([t_a_emb, model.encode_text(tokens)])

    return t_n_emb, t_a_emb

class TextInceptionSegDecoder(nn.Module):
    def __init__(self, prompt_templates, model, output_dim, 
            embed_size=768, n_head=8, n_queries=16, img_size=336, patch_size=14):
        super(TextInceptionSegDecoder, self).__init__()

        # For EVA
        # self.extra_encoder, _, _ = create_model_and_transforms("EVA02-CLIP-L-14-336", "pretrained/EVA02_CLIP_L_336_psz14_s6B.pt", force_custom_clip=True)
        
        # For DinoV2
        from transformers import AutoImageProcessor, AutoModel
        cache_dir='/chencanyu-shcdt/huggingface_hub/dinov2-giant'
        self.extra_encoder = AutoModel.from_pretrained(cache_dir)
        self.extra_processor = AutoImageProcessor.from_pretrained(cache_dir)

        self.maskclip_proj = nn.Conv2d(1024, embed_size, kernel_size=1, padding=0, stride=1)
        # self.maskclip_proj = nn.Conv2d(1536, embed_size, kernel_size=1, padding=0, stride=1)
        self.ada_conv = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1)

        # self.proj_dino = nn.Linear(1024, embed_size) # 1024 -> 768
        # self.deconv_3 = nn.ConvTranspose2d(in_channels=embed_size, out_channels=embed_size, kernel_size=3, stride=3, padding=0) # 16 -> 48
        # self.deconv_2 = nn.ConvTranspose2d(in_channels=embed_size, out_channels=embed_size, kernel_size=2, stride=2, padding=0) # 24 -> 48
        # self.deconv_2_1 = nn.ConvTranspose2d(in_channels=embed_size, out_channels=embed_size, kernel_size=2, stride=2, padding=0) # 48 -> 96
        # self.act = QuickGELU()

        self.t_n_emb, self.t_a_emb = build_text_kv(prompt_templates, model)
        self.query_n = self.t_n_emb.shape[0]
        self.query_a = self.t_a_emb.shape[0]
        self.img_size = img_size
        self.patch_size = patch_size
        # self.query_n = 16
        # self.query_a = 16
        self.attn_pool_n = AttentionalPooler(output_dim, embed_size, n_head, self.query_n)
        self.attn_pool_a = AttentionalPooler(output_dim, embed_size, n_head, self.query_a)
        # self.ffn = GELUFFN(in_features=embed_size, hidden_features=8)
        self.ffn = GELUFFN(in_features=embed_size, hidden_features=int(embed_size * 4.))
        self.dropout = nn.Dropout(0.05)
        init_logit_scale = 4.6052
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale, requires_grad=True)
        n_quries = self.query_n + self.query_a
        self.linear1 = nn.Linear(n_queries, 2)
        # self.proj_n = nn.Linear(n_queries, 1)
        # self.proj_a = nn.Linear(n_queries, 1)
        self.n_queries = n_queries
        self.embed_size = embed_size

        self.wrapper_mask_decoder = WrapperMaskDecoder()
        # self.sam_proj_n = nn.Linear(self.embed_size, self.wrapper_mask_decoder.prompt_embed_dim)
        self.sam_proj_n = GELUFFN(in_features=embed_size, hidden_features=8, out_features=self.wrapper_mask_decoder.prompt_embed_dim)
        # self.sam_proj_a = nn.Linear(self.embed_size, self.wrapper_mask_decoder.prompt_embed_dim)
        self.sam_proj_a = GELUFFN(in_features=embed_size, hidden_features=8, out_features=self.wrapper_mask_decoder.prompt_embed_dim)

        self.init_parameters()
    
    def init_parameters(self, sam_r_path='/chencanyu-shcdt/huggingface_hub/sam_vit_l_0b3195.pth'):
        # use prompt_templates to initial k
        # prompt_templates be like [['a crop photo of perfect xxx', 'a small photo of good xxx', ...], 
        #                           ['a crop photo of defect xxx', 'a small photo of bad xxx', ...]]
        # use text encoder to build 
        with open(sam_r_path, "rb") as f:
            if not hasattr(self, 'wrapper_mask_decoder'):
                return
            checkpoint = torch.load(f)
            mask_decoder_state_dict = {k.replace('mask_decoder.', ''): v for k, v in checkpoint.items() if 'mask_decoder' in k}
            self.wrapper_mask_decoder.mask_decoder.load_state_dict(mask_decoder_state_dict, strict=False)


    def extern_forward(self, x, pe):
        '''
        x: [B, N, L]
        pe: [B, N, L]
        '''
        if not hasattr(self, 'wrapper_mask_decoder'):
            return

        device = x.device
        prompt_embed_dim = self.wrapper_mask_decoder.prompt_embed_dim
        side = self.img_size // self.patch_size
        x = x.reshape(x.shape[0], side, side, -1) # x -> [B, N_h, N_w, L]

        self.t_n_emb = self.t_n_emb.to(device)
        self.t_a_emb = self.t_a_emb.to(device)
        kv_n = self.t_n_emb.unsqueeze(0) + self.attn_pool_n(self.t_n_emb.unsqueeze(0)) # [1, n_lens, embed_size] [1, n_lens, embed_size]
        kv_a = self.t_a_emb.unsqueeze(0) + self.attn_pool_a(self.t_a_emb.unsqueeze(0))

        kv_n = self.sam_proj_n(kv_n)
        kv_n = kv_n / kv_n.norm(dim=-1, keepdim=True)

        kv_a = self.sam_proj_a(kv_a)
        kv_a = kv_a / kv_a.norm(dim=-1, keepdim=True)
        
        sim_n = self.wrapper_mask_decoder(x, pe, kv_n)[0] # [B, 1, mask_h, mask_w] mask_h and mask_w maybe side * 4 = 96
        sim_a = self.wrapper_mask_decoder(x, pe, kv_a)[0]

        sim_n = F.interpolate(sim_n, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        sim_a = F.interpolate(sim_a, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        sim = torch.cat([sim_n, sim_a], dim=-1)
        return sim

    @torch.no_grad
    def get_corrs(self, impaths, device, B):
        from PIL import Image
        images = [Image.open(path) for path in impaths]
        inputs = self.extra_processor(images=images, return_tensors="pt").to(device)
        outputs = self.extra_encoder(**inputs)
        last_hidden_state = outputs.last_hidden_state[:, 1:] # [Bs, grid_count=256, embed_size=1024]
        dino_embed = last_hidden_state
        dino_side = int(math.sqrt(dino_embed.shape[1]))
        corrs = torch.bmm(dino_embed, dino_embed.transpose(2, 1))
        corrs = corrs.reshape(B, dino_side, dino_side, dino_side * dino_side).permute(0, 3, 1, 2)
        return corrs # [Bs, h * w, h, w]


    def compute_weighted_pool(self, maskclip_feats: torch.Tensor, corrs: torch.Tensor):
        """
        Weighted pooling method.
        :param maskclip_feats: torch.tensor - raw clip features
        :param corrs: torch.tensor - correlations as weights for pooling mechanism
        :return: torch.tensor - refined clip features
        """
        B = maskclip_feats.shape[0]
        h_m, w_m = maskclip_feats.shape[-2:]
        h_w, w_w = corrs.shape[-2:]

        maskclip_feats_ref = torch.einsum("bnij, bcij -> bcn", corrs, maskclip_feats)  # B C HW
        norm_factor = corrs.flatten(-2, -1).sum(dim=-1)[:, None]  # B 1 HW
        maskclip_feats_ref = maskclip_feats_ref / (norm_factor + 1e-6)

        # RESHAPE back to 2d
        maskclip_feats_ref = maskclip_feats_ref.reshape(B, -1, h_m, w_m)
        return maskclip_feats_ref # [Bs, C, h_m, w_m]

    def forward(self, x, image=None, train=False): 
        res = dict()
        # x as patch token [Bs, grid_count, embed_size]
        B = x.shape[0]
        L = x.shape[2]
        device = x.device

        # Test Dinoiser
        if image is not None: 
            # image as impaths
            corrs = self.get_corrs(image, device, B)
            res['A_ksi'] = corrs.clone()
            dino_side = corrs.shape[2]
            # resize x
            side = int(math.sqrt(x.shape[1]))
            x = x.view(B, side, side, L).permute(0, 3, 1, 2) # [Bs, 1024, 24, 24]
            x = self.maskclip_proj(x) # [Bs, 768, 24, 24]
            # print(f'{x.shape} after maskclip proj')
            x = self.ada_conv(x) # [Bs, 768, 24, 24]
            L = x.shape[1]
            # print(f'{L} L after ada conv')
            # x = x.permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(dino_side, dino_side), mode='bilinear', align_corners=True) # [Bs, embed_size=768, h=16, w=16]
            a = x.clone().permute(0, 2, 3, 1).reshape(B, dino_side * dino_side, L)
            res['A_phi'] = torch.bmm(a, a.transpose(1, 2)).reshape(B, dino_side, dino_side, dino_side * dino_side).permute(0, 3, 1, 2)
            x = self.compute_weighted_pool(x, corrs)
            x = x.permute(0, 2, 3, 1).view(B, dino_side * dino_side, L) # [Bs, new_grid_count=256, embed_size=768]

        ## Test DeConv: Useless

        # side = int(self.img_size // self.patch_size)
        # x = x.view(B, L, side, side)
        # x = self.deconv_2(x)
        # x = self.act(x)
        # x = self.deconv_2_1(x)
        # x = x.view(B, x.shape[2] * x.shape[3], L) # [Bs, 96*96, embed_size]


        # for MOF EVA, image as tensor
        # if image is not None: 
        #     output = self.extra_encoder.encode_image(image)
        #     x = torch.stack((x, output['tokens']), dim=2).view(x.shape[0], 2 * x.shape[1], x.shape[2])
        #     x = output['tokens']

        # for MOF DINOv2, image as list of path
        # if image is not None: 
        #     # image as ['', '']

        #     from PIL import Image
        #     import math
        #     images = [Image.open(path) for path in image]
        #     inputs = self.extra_processor(images=images, return_tensors="pt").to(device)
        #     outputs = self.extra_encoder(**inputs)
        #     last_hidden_state = outputs.last_hidden_state[:, 1:] # [Bs, grid_count=256, embed_size=1024]
        #     dino_embed = self.proj_dino(last_hidden_state) # [Bs, grid_count=256, embed_size=768]
        #     dino_side = int(math.sqrt(dino_embed.shape[1]))
        #     dino_embed = dino_embed.view(B, L, dino_side, dino_side) # [Bs, embed_size=768, 16, 16]
        #     dino_embed = self.deconv_3(dino_embed) # [Bs, embed_size, 48, 48]
        #     dino_embed = dino_embed.view(B, dino_embed.shape[2] * dino_embed.shape[3], L) # [Bs, 48*48, embed_size]

        #     side = int(self.img_size // self.patch_size)
        #     x = x.view(B, L, side, side)
        #     x = self.deconv_2(x)
        #     x = x.view(B, x.shape[2] * x.shape[3], L) # [Bs, 48*48, embed_size]

        #     x = torch.stack((x, dino_embed), dim=2).view(B, 2 * x.shape[1], L)

        
        self.t_n_emb = self.t_n_emb.to(device)
        self.t_a_emb = self.t_a_emb.to(device)
    
        # Attn Pool

        kv_n = self.t_n_emb.unsqueeze(0) + self.attn_pool_n(self.t_n_emb.unsqueeze(0)) # [1, n_lens, embed_size] [1, n_lens, embed_size]
        kv_a = self.t_a_emb.unsqueeze(0) + self.attn_pool_a(self.t_a_emb.unsqueeze(0))
        # x = x + self.dropout(self.ffn(x))
        # x = self.transformer(x, x)
        x = x / x.norm(dim=-1, keepdim=True)
        kv_t = torch.cat([kv_n, kv_a], dim=1)
        kv_t = kv_t / kv_t.norm(dim=-1, keepdim=True)
        feats = torch.bmm(x, kv_t.expand(x.size(0), kv_t.size(1), kv_t.size(2)).transpose(1, 2)) # [Bs, grid_count, n_quries * 2]
        feats_n = torch.mean(feats[:, :, :self.query_n], dim=2)
        feats_a = torch.mean(feats[:, :, self.query_a:], dim=2)
        # feats_n = self.proj_n(feats[:, :, :self.n_queries])
        # feats_a = self.proj_a(feats[:, :, self.n_queries:])
        feats = torch.cat([feats_n.unsqueeze(2), feats_a.unsqueeze(2)], dim=2)
        # feats = self.linear1(feats) # [Bs, grid_count, 2]
        sim = feats * self.logit_scale.exp() # [Bs, grid_count, 2]
        
        # origin
        # x = x / x.norm(dim=-1, keepdim=True)
        # kv_t = torch.cat([self.t_n_emb.mean(dim=0).unsqueeze(0), self.t_a_emb.mean(dim=0).unsqueeze(0)], dim=0) # [2, L]
        # kv_t = kv_t.unsqueeze(0)
        # kv_t = kv_t / kv_t.norm(dim=-1, keepdim=True)
        # feats = torch.bmm(x, kv_t.expand(x.size(0), kv_t.size(1), kv_t.size(2)).transpose(1, 2))
        # sim = feats * self.logit_scale.exp()


        num_patches = x.size(1)
        # if image is not None: # for MOF
        #     sim = torch.mean(sim.view(sim.shape[0], sim.shape[1]//2, 2, sim.shape[2]), dim=2)

        sim = sim[:, :num_patches, :]  # sim = [batch_size, num_patches, 2]
        # side = int(self.img_size // self.patch_size)  # side
        side = int(math.sqrt(num_patches))
        sim = sim.reshape(sim.shape[0], side, side, -1).permute(0, 3, 1, 2)
        sim = torch.nn.functional.interpolate(sim, self.img_size, mode='bilinear')
        sim = sim.permute(0, 2, 3, 1)

        if train:
            return sim, res

        return sim # [B, image_H, image_W, 2]

class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_size):
        super(CrossAttentionPooling, self).__init__()
        # Cross-attention layers
        self.query_proj = nn.Linear(embed_size, embed_size)
        self.key_proj = nn.Linear(embed_size, embed_size)
        self.value_proj = nn.Linear(embed_size, embed_size)
        self.output_proj = nn.Linear(embed_size, embed_size)

    def forward(self, text_features, image_cls_token):
        # Broadcast image_cls_token to match text_features shape
        image_cls_token = image_cls_token.unsqueeze(1).expand(-1, text_features.size(0), -1)

        # Project query, key, and value
        Q = self.query_proj(text_features)  # Query: text features [n_text, embed_size]
        K = self.key_proj(image_cls_token)  # Key: image cls token [batch_size, n_text, embed_size]
        V = self.value_proj(image_cls_token)  # Value: image cls token [batch_size, n_text, embed_size]

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute pooled text token
        pooled_text_token = torch.matmul(attention_weights, V)  # Shape: [batch_size, n_text, embed_size]

        # Pooling: Reduce [batch_size, n_text, embed_size] to [batch_size, embed_size]
        pooled_text_token = pooled_text_token.mean(dim=1)  # Average pooling over n_text
        pooled_text_token = self.output_proj(pooled_text_token)

        return pooled_text_token

class CrossInceptionSegDecoder(nn.Module):
    def __init__(self, prompt_templates, model, output_dim, 
            embed_size=768, n_head=8, n_queries=16, img_size=336, patch_size=14):
        super(CrossInceptionSegDecoder, self).__init__()
        self.t_n_emb, self.t_a_emb = build_text_kv(prompt_templates, model)
        self.query_n = self.t_n_emb.shape[0]
        self.query_a = self.t_a_emb.shape[0]
        self.img_size = img_size
        self.patch_size = patch_size
        self.attn_pool_n = AttentionalPooler(output_dim, embed_size, n_head, self.query_n)
        self.attn_pool_a = AttentionalPooler(output_dim, embed_size, n_head, self.query_a)

        self.cross_pool_n = CrossAttentionPooling(embed_size=embed_size)
        self.cross_pool_a = CrossAttentionPooling(embed_size=embed_size)

        self.ffn = GELUFFN(in_features=embed_size, hidden_features=int(embed_size * 4.))
        self.dropout = nn.Dropout(0.05)
        init_logit_scale = 4.6052
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale, requires_grad=True)
        n_quries = self.query_n + self.query_a
        self.linear1 = nn.Linear(n_queries, 2)
        self.gaussian_blur = GaussianBlur()
        self.n_queries = n_queries
        self.embed_size = embed_size
    
    def init_parameters(self, sam_r_path='/chencanyu-shcdt/huggingface_hub/sam_vit_l_0b3195.pth'):
        pass

    def forward(self, x, cls_token, image=None, train=False): 
        res = dict()
        # x as patch token [Bs, grid_count, embed_size]
        # cls_token as class token [Bs, embed_size]
        B = x.shape[0]
        L = x.shape[2]
        device = x.device
        
        self.t_n_emb = self.t_n_emb.to(device)
        self.t_a_emb = self.t_a_emb.to(device)

        pool_tn = self.cross_pool_n(self.t_n_emb, cls_token) # [bs, embed_size]
        pool_ta = self.cross_pool_a(self.t_a_emb, cls_token) # [bs, embed_size]

        pool_t = torch.cat([pool_tn.unsqueeze(1), pool_ta.unsqueeze(1)], dim=1) # [bs, 2, embed_size]
        pool_t = pool_t / pool_t.norm(dim=-1, keepdim=True)
        x = x / x.norm(dim=-1, keepdim=True)
        feats = torch.bmm(x, pool_t.transpose(1, 2)) # [Bs, grid_count, 2]
        sim = feats * self.logit_scale.exp() # [Bs, grid_count, 2]
        
        # origin
        # x = x / x.norm(dim=-1, keepdim=True)
        # kv_t = torch.cat([self.t_n_emb.mean(dim=0).unsqueeze(0), self.t_a_emb.mean(dim=0).unsqueeze(0)], dim=0) # [2, L]
        # kv_t = kv_t.unsqueeze(0)
        # kv_t = kv_t / kv_t.norm(dim=-1, keepdim=True)
        # feats = torch.bmm(x, kv_t.expand(x.size(0), kv_t.size(1), kv_t.size(2)).transpose(1, 2))
        # sim = feats * self.logit_scale.exp()


        num_patches = x.size(1)
        # if image is not None: # for MOF
        #     sim = torch.mean(sim.view(sim.shape[0], sim.shape[1]//2, 2, sim.shape[2]), dim=2)

        sim = sim[:, :num_patches, :]  # sim = [batch_size, num_patches, 2]
        # side = int(self.img_size // self.patch_size)  # side
        side = int(math.sqrt(num_patches))
        sim = sim.reshape(sim.shape[0], side, side, -1).permute(0, 3, 1, 2) # [Bs, 2, side_H, side_W]
        sim = self.gaussian_blur(sim)
        sim = torch.nn.functional.interpolate(sim, self.img_size, mode='bilinear')
        sim = sim.permute(0, 2, 3, 1)

        if train:
            return sim, res

        return sim # [B, image_H, image_W, 2]

class SegDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, embed_size=1024, output_size=768, size=2, num_patches=576):
        super(SegDecoder, self).__init__()
        # self.attention = PatchAttention()
        self.embed_size = embed_size
        self.output_size = output_size
        # self.conv1 = nn.Conv2d(in_channels=embed_size, out_channels=embed_size, kernel_size=3, stride=1, padding=1)
        self.transformer = nn.ModuleList([TransformerDecoder(d_model=self.embed_size, nhead=int(self.embed_size//64)) for _ in range(size)])

        scale = embed_size ** -0.5
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(num_patches, embed_size)
        )

        # self.group_vit = GroupViT(input_dim, output_dim)
        # self.list = nn.ModuleList([FFN(embed_size) for _ in range(size)])
        scale = embed_size ** -0.5
        self.proj = nn.ParameterList([nn.Parameter(scale * torch.randn(embed_size, output_size)) for _ in range(size)])

    def inject_proj(self, proj):
        for i in self.proj:
            i.copy_(proj.data)

    def forward(self, x, t, idx=-1):
        # grid_size = 336 // 14
        # x = x[:, :grid_size * grid_size, :]
        # x = x.permute(0, 2, 1)
        # x = x.reshape(x.shape[0], self.embed_size, grid_size, grid_size)
        # x = self.conv1(x)
        # x = x.reshape(x.shape[0], self.embed_size, grid_size * grid_size)
        # x = x.permute(0, 2, 1)
        x = x + self.positional_embedding
        x = self.transformer[idx](x, t)
        x = x @ self.proj[idx]

        # x = self.attention(x)  # [32, 576, 786*2]
        # x = self.list[idx](x)  # [32, 576, 786]
        # x = self.proj[idx](x.permute(0, 2, 1)).permute(0, 2, 1) # [32, 592, 786]
        # return x.permute(0, 2, 1)
        return x


class FPNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

class FPN(nn.Module):
    def __init__(self, embed_size, num_layers=4, channels=[256, 256, 256, 256]):
        super(FPN, self).__init__()
        self.num_layers = num_layers
        self.channels = channels
        
        # 初始化FPN层
        self.fpn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.fpn_layers.append(FPNLayer(embed_size, channels[i]))
        
        # 初始化上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # 初始化下采样层
        self.downsample = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        # 输入形状: [B, grid_size, embed_size]
        features = []
        
        # 第一层FPN
        x = self.fpn_layers[0](x)
        features.append(x)
        
        # 后续层FPN
        for i in range(1, self.num_layers):
            # 下采样
            x = self.downsample(x)
            x = self.fpn_layers[i](x)
            features.append(x)
        
        # 上采样并融合特征
        for i in range(self.num_layers - 2, -1, -1):
            features[i] = features[i] + self.upsample(features[i + 1])
        
        return features