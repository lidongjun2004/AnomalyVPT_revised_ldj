import os

import loguru
import torch
from tqdm import tqdm

from src.datasets.DataLoader import DataManager
from src.helper import build_model, load_checkpoint
from src.utils.metrics import AdEvaluator


@torch.no_grad()
def test(cfg, device):
    dm = DataManager(cfg)
    test_loader = dm.test_loader
    latest_path = os.path.join(cfg.OUTPUT_DIR, 'model-latest.pth.tar')
    model = build_model(cfg, device, is_train=False)[0]
    evaluator = AdEvaluator(cfg, device)

    if cfg.RESUME.__len__() > 0:
        load_path = cfg.RESUME
    elif os.path.exists(latest_path):
        load_path = latest_path
    else:
        loguru.logger.error("Could not find weights path")
        raise ValueError("Could not find weights path")

    model = load_checkpoint(r_path=load_path, device=device, model=model)[0]

    model.eval()
    evaluator.reset()
    model.float()

    up = cfg.PIXEL
    vis = cfg.VIS
    for (idx, batch) in tqdm(enumerate(test_loader), desc="Test Data"):
        image, label, mask, reality_names, mask_paths, impaths = parse_batch(cfg, batch, device)
        res = model(image, is_train=False, up=up, impaths=impaths)
        output = res['logits']
        evaluator.add_score_and_label(output, label, reality_names)

        if up:
            score_map = res['out_map'].softmax(-1)[..., 1]
            evaluator.add_map_and_mask(score_map, mask, mask_paths, impaths, reality_names)

        del image, mask

    evaluator.evaluate()
    if vis:
        evaluator.vis_all(length=20, norm_type='img')


def parse_batch(cfg, batch, device):
    image = batch["img"]
    label = batch["label"]
    mask = batch["mask"]
    image = image.to(device)
    label = label.to(device)
    mask = mask.to(device)
    reality_names = batch["reality_name"]
    impaths = batch["impath"]
    mask_paths = batch["mask_path"]
    return image, label, mask, reality_names, mask_paths, impaths
