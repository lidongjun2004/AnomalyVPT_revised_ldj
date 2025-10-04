import os

import loguru
import torch

from src.open_clip import create_customer_model_and_transforms
from src.eva_clip import create_model_and_transforms
from src.models.CustomCLIP_CATSeg import CustomCLIP
from src.models.CustomEVACLIP import CustomEVACLIP


_PRETRAINED = {
    "EVA02-CLIP-L-14-336": "pretrained/EVA02_CLIP_L_336_psz14_s6B.pt",
    "EVA01-CLIP-g-14-plus": "pretrained/EVA01_CLIP_g_14_plus_psz14_s11B.pt", 
    "ViT-L-14-336": "pretrained/ViT-L-14-336px.pt",
    "ViT-B-16": "pretrained/ViT-B-16px.pt"
}

def build_model(cfg, device, is_train=True):
    model_name = cfg.MODEL.MODEL_NAME
    loguru.logger.info(f'Loading model {model_name}')
    pretrained = _PRETRAINED[model_name]
    
    loguru.logger.info(f'Building CustomCLIP')
    if model_name.find("EVA") != -1:
        loguru.logger.info("Now load eva model")
        clip_model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
        model = CustomEVACLIP(cfg, clip_model)
    else:
        loguru.logger.info("Now load openai model")
        clip_model, _, preprocess = create_customer_model_and_transforms(model_name, pretrained='openai')
        model = CustomCLIP(cfg, clip_model)
    opt = None
    scheduler = None

    if is_train:
        # learnable_params = ["learnable_prompt"]
        # learnable_params = ["fpn_decoder"]
        # learnable_params = ["segti_decoder"]
        learnable_params = ["seg_decoder", "learnable_prompt"]
        # learnable_params = ["cross_decoder"]
        excluded_params = ["wrapper_mask_decoder", "extra_encoder"]
        # learnable_params = ["fpn1", "fpn2", "fpn3", "fpn4", "fpn_decoder"]
        # params_groups = torch.nn.ParameterList()
        params_groups = []

        lr = cfg.OPTIM.LR
        weight_decay = cfg.OPTIM.WEIGHT_DECAY
        adam_beta1 = cfg.OPTIM.ADAM_BETA1
        adam_beta2 = cfg.OPTIM.ADAM_BETA2

        # decay_params_group = {
        #     'params': torch.nn.ParameterList()
        #     'weight_decay': weight_decay,
        #     'betas=(adam_beta1, adam_beta2)'
        # }
    
        # 遍历模型的参数
        for name, param in model.named_parameters():
            # 检查是否在 excluded_params 中
            if any(excluded in name for excluded in excluded_params):
                param.requires_grad_(False)
                # loguru.logger.info(f"{name} is excluded and does not require grad.")
            else:
                # 检查是否在 learnable_params 中
                if any(learnable in name for learnable in learnable_params):
                    param.requires_grad_(True)
                    loguru.logger.success(f"{name} requires grad.")
                    # params_groups.append(param)

                    if ('bias' in name) or (len(param.shape) == 1):
                        params_groups.append({'params': param, 'WD_exclude': True, 'weight_decay': 0.0})
                    else:
                        params_groups.append({'params': param, 'weight_decay': weight_decay})
                else:
                    param.requires_grad_(False)
                    # loguru.logger.info(f"{name} does not require grad.")

        opt = torch.optim.AdamW(params_groups,
                                lr=lr,
                                # weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2),
                                )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.OPTIM.MAX_EPOCH)

    model.to(device)
    return model, opt, scheduler


def load_checkpoint(r_path, device, model, opt=None):
    try:
        checkpoint = torch.load(r_path, map_location=device)
        loguru.logger.success(f"loaded checkpoint from {r_path}")
        epoch = checkpoint['epoch']

        # -- loading model
        pretrained_dict = checkpoint['model']
        msg = model.load_state_dict(pretrained_dict, strict=False)
        loguru.logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        if opt is not None:
            opt.load_state_dict(checkpoint['opt'])

        loguru.logger.info(f'loaded optimizers from epoch {epoch}')
        loguru.logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        loguru.logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return model, opt, epoch
