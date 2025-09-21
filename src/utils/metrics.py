import os

import loguru
import numpy as np
import torch
from tabulate import tabulate
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryAveragePrecision
from tqdm import tqdm
from torchvision.transforms import transforms
from PIL import Image
from .visualize import visualizer
from .aupro import AUPRO
import torch.nn.functional as F
import torch.nn as nn


def compute_auroc(output, target):
    pred = output.softmax(dim=-1)[:, 1]
    # return roc_auc_score(target.cpu().numpy(), pred.cpu().numpy()) * 100
    auroc = BinaryAUROC()
    return auroc(pred, target) * 100


def compute_p_auroc(output, target):
    pred = output
    auroc = BinaryAUROC()
    return auroc(pred, target) * 100


class GaussianFilter(nn.Module):
    def __init__(self, size, sigma, device):
        super(GaussianFilter, self).__init__()
        self.device = device
        self.size = size
        self.sigma = sigma
        self.padding = size // 2
        self.register_buffer('kernel', self._gaussian_kernel().to(self.device))

    def _gaussian_kernel(self):
        kernel = torch.exp(-(torch.arange(self.size) -
                             self.padding) ** 2 / (2 * self.sigma ** 2))
        return kernel / kernel.sum()

    def forward(self, x):
        x = F.conv2d(x.unsqueeze(1), self.kernel.view(
            1, 1, -1, 1), padding=(self.padding, 0))
        x = F.conv2d(x, self.kernel.view(1, 1, 1, -1),
                     padding=(0, self.padding))
        return x.squeeze(1)


class AdEvaluator:
    def __init__(self, cfg, device):
        # if 'ASCEND_HOME_PATH' in os.environ:
        #     self.device = torch.device('cpu')
        # else:
        #     self.device = device
        self.device = device
        self.cfg = cfg
        self.img_size = cfg.INPUT.SIZE[0]
        # self.img_size = 448
        self.gaussian_filler = GaussianFilter(size=5, sigma=4, device=device)
        self.best_i_auroc = 0
        self.best_p_auroc = 0
        # self.gt = {}
        # self.score = {}
        # self.reality_names = {}
        # self.table = []
        # self.i_auroc = []
        # self.i_acc = []
        # self.i_tnr = []
        # self.i_mean_score = []
        # self.i_aupr = []

        # self.mask = {}
        # self.mask_paths = {}
        # self.img_paths = {}
        # self.score_map = {}
        # self.p_auroc = []
        # self.p_aupro = []

    '''
    score: Tensor(bs,)
    label: Tensor(bs,)
    reality_name: Dict(str)
    mask: Dict(mask_path)
    '''

    def reset(self):
        self.gt = {}
        self.score = {}
        self.reality_names = {}
        self.table = []
        self.i_auroc = []
        self.i_acc = []
        self.i_tnr = []
        self.i_mean_score = []
        self.i_aupr = []

        self.mask = {}
        self.mask_paths = {}
        self.img_paths = {}
        self.score_map = {}
        self.p_auroc = []
        self.p_aupro = []

    def add_score_and_label(self, score, label, reality_name):
        # score = score[:, 1]
        device = self.device
        label = label.to(device)
        score = score.to(device)
        
        score = score.softmax(dim=-1)[:, 1]
        for idx, name in enumerate(reality_name):
            if name not in self.reality_names:
                self.reality_names[name] = name
                self.gt[name] = torch.empty(0, device=device)
                self.score[name] = torch.empty(0, device=device)
            cur_label = label[idx].unsqueeze(0).float()
            cur_score = score[idx].unsqueeze(0)
            self.gt[name] = torch.cat((self.gt[name], cur_label))
            self.score[name] = torch.cat((self.score[name], cur_score))

    '''
    score_map: [bs, h, w]
    mask_paths: List[str]
    '''

    def add_map_and_mask(self, score_map, mask, mask_paths, impaths, reality_name):
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        mask = mask.to(self.device)
        score_map = score_map.to(self.device)
        for idx, name in enumerate(reality_name):
            if self.mask.get(name, None) is None:
                self.mask[name] = torch.empty(0).to(self.device)
                self.score_map[name] = torch.empty(0).to(self.device)
                self.mask_paths[name] = []
                self.img_paths[name] = []
            cur_mask_path = mask_paths[idx]
            cur_mask = mask[idx]
            self.mask_paths[name].append(cur_mask_path)
            self.img_paths[name].append(impaths[idx])
            # cur_score_map = self.gaussian_filler(score_map[idx].unsqueeze(0))
            cur_score_map = score_map[idx].unsqueeze(0)
            self.mask[name] = torch.cat((self.mask[name], cur_mask), dim=0)
            self.score_map[name] = torch.cat((self.score_map[name], cur_score_map), dim=0)

    def evaluate(self):
        for idx, name in tqdm(enumerate(self.reality_names), desc="Evaluating:"):
            # device = self.score[name].device
            auroc = BinaryAUROC().to(self.device)
            ap = BinaryAveragePrecision().to(self.device)
            pro = AUPRO().to(self.device)
            pred = self.score[name]
            target = self.gt[name]
            pred_px = self.score_map.get(name, None)
            target_px = self.mask.get(name, None)

            i_auroc = auroc(pred, target)
            i_auroc = i_auroc.item() * 100
            self.i_auroc.append(i_auroc)

            i_aupr = ap(pred, target.long())
            i_aupr = i_aupr.item() * 100
            self.i_aupr.append(i_aupr)

            if pred_px is not None:
                p_auroc = auroc(pred_px, target_px)
                p_auroc = p_auroc.item() * 100
                # TODO: 改成正常的AUPRO
                p_aupro = -1
                # p_aupro = pro(pred_px, target_px.long())
                # p_aupro = p_aupro.item() * 100
            else:
                p_auroc = -1
                p_aupro = -1
            self.p_auroc.append(p_auroc)
            self.p_aupro.append(p_aupro)

            acc = BinaryAccuracy().to(self.device)
            i_acc = acc(pred, target).item() * 100
            self.i_acc.append(i_acc)

            i_tnr = torch.sum(target == 1).item() / (
                    torch.sum(target == 0).item() + torch.sum(target == 1).item()) * 100
            self.i_tnr.append(i_tnr)

            i_mean_score = pred.mean().item() * 100
            self.i_mean_score.append(i_mean_score)

            # line = [name, i_auroc, i_aupr, p_auroc, p_aupro]
            line = [name, i_auroc, i_aupr, p_auroc]
            self.table.append(line)

        mean_i_auroc = np.mean(self.i_auroc)
        mean_i_aupr = np.mean(self.i_aupr)
        mean_p_auroc = np.mean(self.p_auroc)
        mean_p_aupro = np.mean(self.p_aupro)
        mean_i_acc = np.mean(self.i_acc)
        # self.table.append(['mean', mean_i_auroc, mean_i_aupr, mean_p_auroc, mean_p_aupro])
        self.table.append(['mean', mean_i_auroc, mean_i_aupr, mean_p_auroc])
        # results = tabulate(self.table, headers=['objects', 'i_auroc', 'i_aupr', 'p_auroc', 'p_aupro'])
        results = tabulate(self.table, headers=['objects', 'i_auroc', 'i_aupr', 'p_auroc'])
        loguru.logger.info('\n' + results)

        if mean_i_auroc > self.best_i_auroc:
            self.best_i_auroc = mean_i_auroc
            return True
        elif mean_p_auroc > self.best_p_auroc:
            self.best_p_auroc = mean_p_auroc
            return True
        else:
            return False

    # def vis(self, reality_name, length=20):
    #     cls_names = [reality_name] * length
    #     mask_paths = self.mask_paths.get(reality_name, None)
    #     save_path = os.path.join(self.cfg.OUTPUT_DIR, 'vis_test')
    #     if mask_paths is not None:
    #         pathes = []
    #         anomaly_map = torch.empty(0).to(self.device)
    #         for idx, path in enumerate(mask_paths):
    #             if path.__len__() > 0:
    #                 pathes.append(path.replace('_mask', '').replace('ground_truth', 'test'))
    #                 anomaly_map = torch.cat([anomaly_map, self.score_map[reality_name][idx].unsqueeze(0)], dim=0)
    #             if pathes.__len__() >= length:
    #                 break
    #         visualizer(pathes, anomaly_map.cpu(), self.img_size, save_path, cls_names[:length])

    def vis_all(self, length=-1, norm_type='img'):
        '''
        norm_type: image-wise or class-wise, 'img' | 'cls'
        '''
        for idx, reality_name in tqdm(enumerate(self.reality_names), desc="Plotting:"):
            cls_names = []
            img_paths = self.img_paths.get(reality_name, None)
            save_path = os.path.join(self.cfg.OUTPUT_DIR, f"vis_test_{norm_type}")
            if img_paths is not None:
                pathes = []
                anomaly_map = torch.empty(0).to(self.device)

                for idx, path in enumerate(img_paths[1+length:]):
                    if path.__len__() > 0:
                        pathes.append(path)
                        anomaly_map = torch.cat([anomaly_map, self.score_map[reality_name][idx].unsqueeze(0)], dim=0)
                        cls_names.append(reality_name)
                visualizer(pathes, anomaly_map.cpu(), self.img_size, save_path, cls_names, norm_type)
