from torch import nn
import torch
import torch.nn.functional as F

from src.open_clip import tokenizer

from src.models.SegDecoder import SegDecoder, FPN, TextInceptionSegDecoder, CrossInceptionSegDecoder, GaussianBlur
from src.models.CATSegDecoder import CATSegDecoder

class PromptOrder:
    def __init__(self) -> None:
        super().__init__()
        self.state_normal_list = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
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
            "a cropped photo of the {}.",
            "a close-up photo of a {}.",
            "a close-up photo of the {}.",
            "a bright photo of a {}.",
            "a bright photo of the {}.",
            "a dark photo of the {}.",
            "a dark photo of a {}.",
            "a jpeg corrupted photo of the {}.",
            "a jpeg corrupted photo of the {}.",
            "a blurry photo of the {}.",
            "a blurry photo of a {}.",
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a small {}.",
            "a photo of the small {}.",
            "a photo of a large {}.",
            "a photo of the large {}.",
            "a photo of the {} for visual inspection.",
            "a photo of a {} for visual inspection.",
            "a photo of the {} for anomaly detection.",
            "a photo of a {} for anomaly detection."
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
        # self.ssl_model =
        embed_size = self.clip_model.text_projection.size(1)
        # self.embed_size = embed_size
        self.dtype = self.clip_model.logit_scale.dtype
        text_prompts = PromptOrder().prompt()
        self.text_features = nn.Parameter(
            self.build_ensemble_text_features(text_prompts, dtype=self.dtype)) # shape = [2, 512]
        self.logit_scale = self.clip_model.logit_scale
        self.ortho_reg = OrthogonalRegularization(
            self.clip_model.visual.learnable_prompt.p)

        # pixel level
        self.image_size = cfg.INPUT.SIZE
        # self.image_size = (448, 448)
        self.patch_size = cfg.MODEL.PATCH_SIZE
        num_patches = (self.image_size[0] // self.patch_size) * \
            (self.image_size[1] // self.patch_size)

        prompt_length = cfg.MODEL.VP_LENGTH
        self.gaussian_blur = GaussianBlur()
        self.seg_decoder = CATSegDecoder(input_dim=num_patches + prompt_length,
                                      output_dim=num_patches + prompt_length,
                                      embed_size=self.clip_model.visual.embed_dim,
                                      output_size=embed_size,
                                      num_patches=num_patches,
                                      size=4)
        # todo: inject proj?

        # self.segti_decoder = TextInceptionSegDecoder(prompt_templates=text_prompts,
        #                                              model=self.clip_model,
        #                                              output_dim=embed_size)

        self.cross_decoder = CrossInceptionSegDecoder(prompt_templates=text_prompts,
                                                      model=self.clip_model,
                                                      output_dim=embed_size)

        from src.models.SegDecoder import QuickGELU, SwiGLU, GELUFFN
        # self.mlp = SwiGLU(in_features=embed_size, hidden_features=8)
        self.mlp = GELUFFN(in_features=embed_size, hidden_features=8)

        self.fpn_scale = [2, 1, 1/2, 1/4]
        self.fpn_decoder = nn.ModuleList([
            # self.mlp
            nn.Sequential(
                nn.Linear(embed_size, 8),
                nn.GELU(),
                nn.Linear(8, embed_size),)
            for _ in self.fpn_scale])
        # TODO: check device!!
        # self.simple_text_encoder = SimpleTextEncoder(clip_model)

    @torch.no_grad()
    def build_ensemble_text_features(self, text_prompts, dtype):
        text_features = torch.empty(0)
        for templates in text_prompts:
            ensemble_text_features = torch.empty(0)
            for prompt in templates:
                tokens = tokenizer.tokenize(prompt)
                cur_embed = self.clip_model.encode_text(
                    tokens).type(dtype=dtype) # shape = [1, 512]
                ensemble_text_features = torch.cat(
                    [ensemble_text_features, cur_embed], dim=0)
            # ensemble_text_features.shape = [num_prompt, 512]
            avg_text_features = torch.mean(
                ensemble_text_features, dim=0, keepdim=True) # shape = [1, 512]
            text_features = torch.cat([text_features, avg_text_features]) # shape = [2, 512]

        return text_features

    def forward(self, image, is_train=False, up=True, impaths=None):
        output = dict()
        # b, c, h, w = image.shape
        text_features = self.text_features / \
            self.text_features.norm(dim=-1, keepdim=True)
        res = self.clip_model.encode_image(image.type(self.dtype),
                                           mask=[],
                                           proj=True,
                                           train=is_train)
        origin_image_features = res['pooled'] # shape = [B, C] = [32, 512]
        output['cls_token'] = res['pooled'] # shape = [B, C] = [32, 512]

        origin_layers_image_feature = res['output_layers'] # shape = [4, B, C] = [4, 32, 512]
        image_features = origin_image_features / \
            origin_image_features.norm(dim=-1, keepdim=True) # shape = [B, C] = [32, 512]
        logit_scale = self.logit_scale.exp()
        raw_score = image_features @ text_features.t() # shape = [B, 2] = [32, 2]
        logits = logit_scale * raw_score # shape = [B, 2] = [32, 2]

        output['logits'] = logits # shape = [B, 2] = [32, 2]

        expand_text_features = text_features.repeat(image_features.shape[0], 1, 1) # shape = [B, 2, C] = [32, 2, 512]
        if up:
            # seg decoder
            patch_features = res['tokens'] # shape = [B, H*W, C] = [32, 196, 512]
            patch_features = self.seg_decoder(
                patch_features, expand_text_features, idx=-1)  # shape = [B, H*W, C] = [32, 196, 512]
            patch_features = patch_features / \
                patch_features.norm(dim=-1, keepdim=True) # shape = [B, H*W, C] = [32, 196, 512]
            anomaly_map = calc_anomaly_map(patch_features, text_features, patch_size=self.patch_size, img_size=self.image_size[0],
                                           scale=logit_scale, blur=self.gaussian_blur)
            
            mid_map = []
            mid_patch_features = res['output_token_layers']
            for idx, mid_patch_feature in enumerate(mid_patch_features):
                mid_patch_feature = self.seg_decoder(
                    mid_patch_feature, expand_text_features, idx)
                mid_patch_feature = mid_patch_feature / \
                    mid_patch_feature.norm(dim=-1, keepdim=True)
                mid_map.append(calc_anomaly_map(mid_patch_feature, text_features, img_size=self.image_size[0],
                                                scale=logit_scale, blur=self.gaussian_blur))

            # # output['out_map'] = harmonic_mean_images(stack_images(anomaly_map, []))

            # # fpn decoder
            # TODO: aggregate output map with different scale, now FPN from the last layer output of ViT
            # fpn_maps = []
            # fpn_features = res['fpn_feat'] # TODO: check device!

            # for idx, fpn_feature in enumerate(fpn_features):
            #     fpn_feature = self.fpn_decoder[idx](fpn_feature) # [B, H*W, embed_size]
            #     fpn_feature = fpn_feature / fpn_feature.norm(dim=-1, keepdim=True)
            #     cur_scale_map = calc_anomaly_map(fpn_feature,
            #                                     text_features,
            #                                     img_size=self.image_size[0],
            #                                     patch_size=self.patch_size // self.fpn_scale[idx])
            #     fpn_maps.append(cur_scale_map) # [bs, pixel_count, 2]
            # output['out_map'] = aggregate_fpn_logits(fpn_maps)
            # print(f"output map shape: {output['out_map'].shape}")

            # segti decoder

            patch_features = res['tokens']
            # img_pe = self.clip_model.visual.positional_embedding[1:, :].unsqueeze(0)
            # img_pe = F.interpolate(img_pe, size=self.clip_model.text_projection.size(1), mode='linear')
            # print(impaths)
            seg_res = None
            # print(f"success enter, impaths len: {len(impaths)}")

            # anomaly_map, seg_res = self.segti_decoder(patch_features, image=impaths, train=True)

            # anomaly_map = self.segti_decoder.extern_forward(patch_features, img_pe)


            # cross decoder
            # anomaly_map = self.cross_decoder(patch_features, origin_image_features)

            output['mid_map'] = []
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
                raw_score = feat @ text_features.t()
                cur_logits = logit_scale * raw_score
                mid_logits.append(cur_logits)
            output['mid_logits'] = mid_logits

        return output
