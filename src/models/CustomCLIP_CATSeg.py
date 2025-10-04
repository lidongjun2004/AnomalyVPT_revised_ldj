from torch import nn
import torch
import torch.nn.functional as F

from einops import rearrange, repeat

from src.open_clip import tokenizer

from src.models.SegDecoder import SegDecoder, FPN, TextInceptionSegDecoder, CrossInceptionSegDecoder, GaussianBlur
from src.models.CATSegDecoder import CATSegDecoder

class PromptOrder:
    def __init__(self) -> None:
        super().__init__()
        self.state_normal_list = [
            "{}",
            # "flawless {}",
            # "perfect {}",
            # "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage"
        ]

        self.state_anomaly_list = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage"
        ]

        self.template_list = [
            "a cropped photo of the {}."
            # "a close-up photo of a {}.",
            # "a close-up photo of the {}.",
            # "a bright photo of a {}.",
            # "a bright photo of the {}.",
            # "a dark photo of the {}.",
            # "a dark photo of a {}.",
            # "a jpeg corrupted photo of the {}.",
            # "a jpeg corrupted photo of the {}.",
            # "a blurry photo of the {}.",
            # "a blurry photo of a {}.",
            # "a photo of a {}.",
            # "a photo of the {}.",
            # "a photo of a small {}.",
            # "a photo of the small {}.",
            # "a photo of a large {}.",
            # "a photo of the large {}.",
            # "a photo of the {} for visual inspection.",
            # "a photo of a {} for visual inspection.",
            # "a photo of the {} for anomaly detection.",
            # "a photo of a {} for anomaly detection."
        ]

    def prompt(self, class_name='object'):
        # object, product, item, target, {none}
        class_state = [ele.format(class_name)
                       for ele in self.state_normal_list]
        normal_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in
                                    self.template_list]
        class_state = [ele.format(class_name)
                       for ele in self.state_anomaly_list]
        anomaly_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in
                                     self.template_list]
        return [normal_ensemble_template, anomaly_ensemble_template]


def harmonic_mean_images(images, eps=1e-7):
    print(f"images shape: {images.shape}")
    reciprocal_channels = 1 / (images + eps)
    mean_reciprocal = torch.mean(
        reciprocal_channels, dim=1, keepdim=True)
    harmonic_mean = torch.reciprocal(mean_reciprocal)
    return harmonic_mean.squeeze()  # [bs, h, w]


def harmonic_mean_image_list(image_list, eps=1e-7):
    '''
        image_list: [n, B, image_H, image_W, 2]
    '''
    harmonic_mean_denominator = torch.sum(1 / (image_list + eps), dim=0)
    n = image_list.shape[0]
    return n / harmonic_mean_denominator  # [n, B, image_H, image_W, 2]


def aggregate_fpn_logits(image_list, eps=1e-7):
    '''
        image_list: list of [B, image_H, image_W, 2]
    '''
    normalized_seg_maps = [F.normalize(
        seg_map, p=1, dim=-1) for seg_map in image_list]
    normalized_seg_maps_stacked = torch.stack(
        normalized_seg_maps)  # [lens, B, image_H, image_W, 2]

    return torch.max(normalized_seg_maps_stacked, dim=0).values


def calc_anomaly_map(image_features, text_features, patch_size=14, img_size=336, scale=1 / 0.07, blur=None):
    num_patches = int((img_size // patch_size) ** 2)
    text_features = text_features.expand(
        image_features.shape[0], 2, image_features.shape[2])  # [B, 2, C]
    feats = torch.bmm(image_features, text_features.transpose(1, 2))
    sim = feats * scale
    sim = sim[:, :num_patches, :]  # sim = [batch_size, num_patches, 2]
    side = int(img_size // patch_size)  # side
    sim = sim.reshape(sim.shape[0], side, side, -1).permute(0, 3, 1, 2)
    if blur is not None:
        sim = blur(sim)
    sim = torch.nn.functional.interpolate(sim, img_size, mode='bilinear')
    sim = sim.permute(0, 2, 3, 1)
    return sim  # [B, image_H, image_W, 2]


class OrthogonalRegularization(nn.Module):
    def __init__(self, weight, decay=0.001):
        super(OrthogonalRegularization, self).__init__()
        self.weight = weight
        self.decay = decay

    def forward(self):
        if isinstance(self.weight, nn.Parameter):
            inner_products = torch.mm(self.weight, self.weight.t())
            identity_matrix = torch.eye(
                inner_products.size(0), device=inner_products.device)
            ortho_loss = torch.norm(inner_products - identity_matrix, p='fro')
        elif isinstance(self.weight, nn.ParameterList):
            ortho_loss = 0
            for param in self.weight:
                inner_products = torch.mm(param, param.t())
                identity_matrix = torch.eye(
                    inner_products.size(0), device=inner_products.device)
                ortho_loss += torch.norm(inner_products -
                                         identity_matrix, p='fro')
        else:
            raise AttributeError("Weight is not nn.Parameter")
        return self.decay * ortho_loss


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.clip_model = clip_model
        output_dim = self.clip_model.visual.output_dim

        self.dtype = self.clip_model.logit_scale.dtype
        text_prompts = PromptOrder().prompt()
        self.text_features = nn.Parameter(
            self.build_ensemble_text_features(text_prompts, dtype=self.dtype)) # shape = [class_num, prompt_num, output_dim] = [2, 4, 512]
        self.logit_scale = self.clip_model.logit_scale
        self.ortho_reg = OrthogonalRegularization(
            self.clip_model.visual.learnable_prompt.p)

        # pixel level
        self.image_size = cfg.INPUT.SIZE
        self.patch_size = cfg.MODEL.PATCH_SIZE
        num_patches = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)

        prompt_length = cfg.MODEL.VP_LENGTH
        self.gaussian_blur = GaussianBlur()

        self.seg_decoder = CATSegDecoder(
            text_guidance_dim = self.text_features.shape[-1],
            appearance_guidance_dim = self.text_features.shape[-1],
            prompt_channel = self.text_features.shape[1],
            num_heads = self.clip_model.transformer.heads,
            num_layers = self.clip_model.transformer.layers,
            feature_resolution = (self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size)
        )

        self.proj_dim = self.clip_model.visual.width
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)

        self.layer_indexes = [3, 7]
        self.layers = []
        def make_hook(idx):
            def hook(m, inp, out):  
                self.layers.append(out)
            return hook

        for l in self.layer_indexes:
            self.clip_model.visual.transformer.resblocks[l].register_forward_hook(make_hook(l))
        # self.cross_decoder = CrossInceptionSegDecoder(prompt_templates=text_prompts,
        #                                               model=self.clip_model,
        #                                               output_dim=output_dim)

        from src.models.SegDecoder import QuickGELU, SwiGLU, GELUFFN

        self.mlp = GELUFFN(in_features=output_dim, hidden_features=8)

        self.fpn_scale = [2, 1, 1/2, 1/4]
        self.fpn_decoder = nn.ModuleList([
            # self.mlp
            nn.Sequential(
                nn.Linear(output_dim, 8),
                nn.GELU(),
                nn.Linear(8, output_dim),)
            for _ in self.fpn_scale])

    @torch.no_grad()
    def build_ensemble_text_features(self, text_prompts, dtype):
        text_features = torch.empty(0)
        for templates in text_prompts:
            ensemble_text_features = torch.empty(0)
            for prompt in templates:
                tokens = tokenizer.tokenize(prompt)
                cur_embed = self.clip_model.encode_text(
                    tokens).type(dtype=dtype) # shape = [output_dim] = [512]
                ensemble_text_features = torch.cat(
                    [ensemble_text_features, cur_embed], dim=0)
            # ensemble_text_features.shape = [prompt_num, output_dim] = [4, 512]
            ensemble_text_features = ensemble_text_features.unsqueeze(0) # shape = [1, prompt_num, output_dim] = [1, 4, 512]
            text_features = torch.cat([text_features, ensemble_text_features], dim=0)

        return text_features # shape = [class_num, prompt_num, output_dim] = [2, 4, 512]

    def forward(self, image, is_train=False, up=True, impaths=None):
        output = dict()
        text_features = self.text_features / \
            self.text_features.norm(dim=-1, keepdim=True) # shape = [class_num, prompt_num, output_dim] = [2, 4, 512]
        self.layers = []
        res = self.clip_model.encode_image(image.type(self.dtype),
                                           mask=[],
                                           proj=True,
                                           train=is_train)
        origin_image_features = res['pooled'] # shape = [B, 1, output_dim] = [32, 1, 512]
        output['cls_token'] = res['pooled'] # shape = [B, 1, C] = [32, 1, 512]

        origin_layers_image_feature = res['output_layers'] # shape = [4, B, 1, output_dim] = [4, 32, 1, 512]
        image_features = origin_image_features / \
            origin_image_features.norm(dim=-1, keepdim=True) # shape = [B, output_dim] = [32, 512]
        logit_scale = self.logit_scale.exp()
        # image_features: [B, output_dim]
        # text_features: [2, 4, output_dim]
        raw_score = torch.einsum('bod,cqd->bcq', image_features.unsqueeze(1), text_features) # [B, 2, 4]
        logits = logit_scale * raw_score # shape = [B, class_num, prompt_num] = [32, 2, 4]

        output['logits'] = logits # shape = [B, class_num] = [32, 2]

        expand_text_features = text_features.unsqueeze(0).repeat(image_features.shape[0], 1, 1, 1) # shape = [B, class_num, prompt_num, output_dim] = [32, 2, 4, 512]
        if up:
            # seg decoder
            patch_features = res['seg_group'] # shape = [B, patch_num, output_dim] = [32, 196, 512]
            # CLIP ViT features for guidance
            res3 = rearrange(patch_features, "B (H W) C -> B C H W", H=14)
            res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=14)
            res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=14)
            res4 = self.upsample1(res4) # shape = [B, 256, 48, 48]
            res5 = self.upsample2(res5) # shape = [B, 128, 96, 96]
            appearance_guidance = [res3, res4, res5]

            patch_features = rearrange(patch_features, "B (h w) c->B c h w", h=14, w=14)
            score_map = self.seg_decoder(patch_features, expand_text_features, appearance_guidance) 
            # shape = [B, class_num, image_H, image_W] = [32, 2, 224, 224]
            anomaly_map = score_map[:, 1, :, :] # [B, image_H, image_W] = [32, 224, 224]

            mid_map = []
            mid_patch_features = res['seg_feat'] # shape = [lens, B, patch_num, output_dim] = [4, 32, 196, 512]
            for idx, mid_patch_feature in enumerate(mid_patch_features):
                mid_patch_feature = rearrange(mid_patch_feature, "B (h w) c->B c h w", h=14, w=14)
                mid_score_map = self.seg_decoder(
                    mid_patch_feature, expand_text_features, appearance_guidance)
                mid_map.append(mid_score_map[:, 1, :, :]) # [B, image_H, image_W] = [32, 224, 224]

            

            patch_features = res['tokens']
            
            seg_res = None
            
            output['mid_map'] = mid_map
            output['map'] = anomaly_map
            output['out_map'] = anomaly_map
            if seg_res:
                output['A_ksi'] = seg_res['A_ksi']
                output['A_phi'] = seg_res['A_phi']

        if is_train:
            output['ortho_loss'] = self.ortho_reg()
            mid_logits = []
            for idx in range(len(origin_layers_image_feature)):
                feat = origin_layers_image_feature[idx]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                raw_score = torch.einsum('bod,cqd->bcq', image_features.unsqueeze(1), text_features)
                cur_logits = logit_scale * raw_score
                mid_logits.append(cur_logits)
            output['mid_logits'] = mid_logits

        return output
