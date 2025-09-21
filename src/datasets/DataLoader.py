import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset as TorchDataset, RandomSampler, SequentialSampler

from torchvision.transforms import InterpolationMode

from src.utils.tools import read_image
from src.datasets.Dataset import build_dataset


def build_sampler(sampler_type, data_source=None):
    if sampler_type == "RandomSampler":
        return RandomSampler(data_source)

    elif sampler_type == "SequentialSampler":
        return SequentialSampler(data_source)


def build_data_loader(
        cfg,
        sampler_type="SequentialSampler",
        data_source=None,
        batch_size=64,
        is_train=True,
        dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        data_source=data_source,
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
            self,
            cfg,
            dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build train_loader
        train_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN.SAMPLER,
            data_source=dataset.train,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader


class DatasetWrapper(TorchDataset):
    def __init__(self, cfg, data_source, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.is_train = is_train
        interp_mode = InterpolationMode.BICUBIC
        self.transform = []

        if self.is_train:
            self.transform += [T.Resize((360, 360), interpolation=interp_mode)]
            # self.transform += [T.Resize((448, 448), interpolation=interp_mode)]
            # self.transform += [T.ColorJitter(brightness=0.48, contrast=0.28)]
            # self.transform += [T.GaussianBlur(kernel_size=3)]
            # self.transform += [T.RandomRotation(15)]

            self.transform += [T.CenterCrop(cfg.INPUT.SIZE)]
        else:
            # self.transform += [T.Resize((448, 448), interpolation=interp_mode)]
            self.transform += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        self.transform += [T.ToTensor()]
        self.transform = T.Compose(self.transform)

        self.norm = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "reality_name": item.reality_name if hasattr(item, 'reality_name') else '',
            "mask_path": item.mask_path if hasattr(item, 'mask_path') else '',
            "impath": item.impath,
            "index": idx
        }

        if output["mask_path"].__len__() > 0:
            mask = read_image(output["mask_path"]).convert('L')
        else:
            mask = Image.fromarray(np.zeros((32, 32)), mode='L')

        img0 = read_image(item.impath)
        img = self.transform(img0)
        img = self.norm(img)
        mask = self.transform(mask)
        output["img"] = img
        output["mask"] = mask

        return output
