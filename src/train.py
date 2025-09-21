import os
from collections import OrderedDict

import loguru
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.DataLoader import DataManager
from .helper import build_model, load_checkpoint
from src.utils.loss import HingeLoss, FocalLoss, BinaryDiceLoss
from src.utils.meters import MetricMeter
from src.utils.metrics import compute_auroc, compute_p_auroc

def lasso_regularization(model, lambda_l1, specific_params):
    if specific_params is not None and specific_params.__len__() > 0:
        l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() if any(param_str in name for param_str in specific_params))
    else:
        l1_norm = sum(p.abs().sum() for name, p in model.named_parameters())
    return lambda_l1 * l1_norm

def train(cfg, device):
    dm = DataManager(cfg)
    train_loader = dm.train_loader
    save_path = os.path.join(cfg.OUTPUT_DIR, 'model-ep{epoch}.pth.tar')
    latest_path = os.path.join(cfg.OUTPUT_DIR, 'model-latest.pth.tar')
    max_epoch = cfg.OPTIM.MAX_EPOCH
    model, opt, scheduler = build_model(cfg, device, is_train=True)
    start_epoch = 0
    lr = cfg.OPTIM.LR

    if cfg.RESUME.__len__() > 0:
        load_path = cfg.RESUME
    elif os.path.exists(latest_path):
        load_path = latest_path
    else:
        load_path = None

    if load_path is not None:
        model, opt, start_epoch = load_checkpoint(r_path=load_path,
                                                  device=device,
                                                  model=model,
                                                  opt=opt)
        lr = opt.param_groups[0]['lr']
    model.float()

    def detect_anomaly(_loss):
        if not torch.isfinite(_loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def forward(model, image, label, mask, impaths):
        up = cfg.PIXEL
        res = model(image, is_train=True, up=up, impaths=impaths)
        output = res['logits']
        mid_output = res['mid_logits']
        loss_hinge = HingeLoss(th=0.8)
        global_loss = loss_hinge(output, label)

        classify_loss = F.cross_entropy(output, label.long())

        mid_loss = torch.tensor(0).float().to(device)
        for j in range(len(mid_output)):
            mid_loss_hinge = HingeLoss(th=0.5)
            mid_loss = mid_loss_hinge(mid_output[j], label)

        i_loss = 3 * global_loss + 0.5 * mid_loss + classify_loss
        loss = i_loss
        # loss = torch.tensor(0.).to(i_loss.device)
        loss_summary = OrderedDict()
        loss_summary['loss'] = loss.item()
        loss_summary['i-auroc'] = compute_auroc(output.detach(), label).item()

        if up:
            anomaly_map_logits = res['out_map']  # [32, 336, 336, 2]
            anomaly_map = anomaly_map_logits.softmax(-1)
            anomaly_map_logits = anomaly_map_logits.permute(0, 3, 1, 2)  # [32, 2, 336, 336]
            anomaly_map = anomaly_map.permute(0, 3, 1, 2)
            # mid_anomaly_map_logits = res['mid_map']
            loss_focal = FocalLoss()
            loss_dice = BinaryDiceLoss()

            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0

            mask = mask.squeeze().detach()

            focal_loss = loss_focal(anomaly_map, mask)
            a_dice_loss = loss_dice(anomaly_map[:, 1, :, :], mask)
            n_dice_loss = loss_dice(anomaly_map[:, 0, :, :], 1 - mask)
            ce_loss = F.cross_entropy(anomaly_map_logits, mask.long())
            local_loss_hinge = HingeLoss(th=0.5, alpha=0.8)
            local_hinge_loss = local_loss_hinge(anomaly_map, mask)
            p_loss = focal_loss + a_dice_loss + n_dice_loss + ce_loss
            
            specific_params = [
                'linear1.weight',
                'linear2.weight'
            ]
            # specific_params = None
            lambda_l1 = 0.05
            r_loss = lasso_regularization(model.fpn_decoder, lambda_l1, specific_params)

            if 'A_ksi' in res and 'A_phi' in res:
                A_ksi = res['A_ksi']
                A_phi = res['A_phi']
                loss_bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
                ssl_loss = loss_bce(A_phi.flatten(-2, -1), (A_ksi.flatten(-2, -1) > 0.2).float())
                loss += ssl_loss


            loss += p_loss
            loss += r_loss
            loss_summary['loss'] = loss.item()
            loss_summary['p-auroc'] = compute_p_auroc(anomaly_map.permute(0, 2, 3, 1)[..., 1].detach(), mask).item()
            loss_summary['focal_loss'] = focal_loss.item()
            loss_summary['a_dice_loss'] = a_dice_loss.item()
            loss_summary['p-ce_loss'] = ce_loss.item()

            if 'A_ksi' in res and 'A_phi' in res:
                loss_summary['ssl_loss'] = ssl_loss.item()

        losses.update(loss_summary)
        info = []
        info += [f"epoch [{epoch + 1}/{max_epoch}]"]
        info += [f"batch [{idx + 1}/{len(train_loader)}]"]
        info += [f"{losses}"]
        info += [f"lr {lr:.4e}"]
        if (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            loguru.logger.info(" ".join(info))

        return loss

    def save_checkpoint(_epoch):
        save_dict = {
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'epoch': _epoch,
        }
        torch.save(save_dict, latest_path)
        checkpoint_freq = cfg.TRAIN.CHECKPOINT_FREQ
        if (_epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=f'{_epoch + 1}'))

    for epoch in range(start_epoch, max_epoch):
        model.train()
        losses = MetricMeter()
        for (idx, batch) in tqdm(enumerate(train_loader)):
            image, label, mask, impaths = parse_batch(cfg, batch, device)
            loss = forward(model, image, label, mask, impaths)
            opt.zero_grad()
            detect_anomaly(loss)
            loss.backward()
            opt.step()
        scheduler.step()
        lr = opt.param_groups[0]['lr']
        save_checkpoint(epoch)


def parse_batch(cfg, batch, device):
    image = batch["img"]
    label = batch["label"]
    mask = batch["mask"]
    
    image = image.to(device)
    label = label.to(device)
    mask = mask.to(device)

    impaths = batch["impath"]
    return image, label, mask, impaths
