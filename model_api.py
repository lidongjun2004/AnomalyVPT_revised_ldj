import os

import cv2
import loguru
import torch
from torch import nn

from src.helper import build_model, load_checkpoint
from src.utils.visualize import visualize_return_img
from configs.default import _C as cfg_default
import torchvision.transforms as transforms
from PIL import Image

def reset_cfg(cfg):
    cfg.EVAL = True
    cfg.PIXEL = True
    cfg.VIS = True
    cfg.DEVICE = 'cuda'


def setup_cfg():
    cfg = cfg_default.clone()
    config_file = os.path.join('AnomalyVPT/configs/vitl14_ep20.yaml')
    cfg.merge_from_file(config_file)
    reset_cfg(cfg)
    cfg.freeze()
    return cfg


class AnomalyVPTModel(nn.Module):
    def __init__(self, device='cpu', train_set='visa'):
        super(AnomalyVPTModel, self).__init__()
        cfg = setup_cfg()
        if train_set == 'visa':
            load_path = 'AnomalyVPT/weights/train-visa-model-latest.pth.tar'
        else:
            load_path = 'AnomalyVPT/weights/train-mvtec-model-latest.pth.tar'

        self.device = torch.device(device)
        model = build_model(cfg, self.device, is_train=False)[0]
        model = load_checkpoint(r_path=load_path, device=device, model=model)[0]
        model.eval()

        self.model = model
        self.up = True
        self.vis = True
        self.img_size = cfg.INPUT.SIZE[0]

        self.transform =transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def detect(self, image):
        '''
            :return
            decision: str,
            result_img: opencv img,
            origin_mask_img: no sence,
            rect_cord: no sence,
        '''
        with torch.no_grad():
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            tensor_image = self.transform(pil_image).unsqueeze(0).to(self.device)
            res = self.model(tensor_image, is_train=False, up=self.up)
            output = res['logits']
            score = output.softmax(dim=-1)[:, 1][0]
            loguru.logger.info("score: {}".format(score))
            if score > 0.9:
                decision = 'ANOMALY'
            else:
                decision = 'GOOD'
            if self.up:
                score_map = res['out_map'].softmax(-1)[..., 1]
            print(f'score_map: {score_map.shape}')
            return_img = visualize_return_img(image, score_map.cpu(), self.img_size)
        return decision, return_img
