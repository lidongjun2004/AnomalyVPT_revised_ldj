import argparse
import time

import loguru
import os

import torch

from src.test import test
from src.train import train
from src.utils.tools import set_random_seed
from configs.default import _C as cfg_default

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_logger(output=None):
    if output is None:
        return

    if output.endswith(".txt") or output.endswith(".log"):
        fpath = output
    else:
        fpath = os.path.join(output, "file.log")

    if os.path.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")
    print(fpath)
    loguru.logger.add(
        sink=os.path.join(fpath),
        level="INFO",
        rotation="10 MB"
    ),


def reset_cfg(cfg, args):
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.name:
        cfg.DATASET.NAME = args.name

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    cfg.EVAL = args.eval
    cfg.PIXEL = args.pixel
    cfg.VIS = args.vis
    cfg.DEVICE = args.device


def setup_cfg(args):
    cfg = cfg_default.clone()

    # From config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # From input arguments
    reset_cfg(cfg, args)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        loguru.logger.info("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if cfg.EVAL:
        test(cfg, device=device)
    else:
        train(cfg, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file', type=str,
        help='name of config file to load',
        default='configs.yaml')
    parser.add_argument(
        '--output-dir', type=str,
        help='output directory',
        default='./output/train_vpt/')
    parser.add_argument(
        '--name', type=str,
        help='dataset name',
        default='mvtec')
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        '--eval', default=False, action='store_true',
        help="evaluate only"
    )
    parser.add_argument(
        '--pixel', default=False, action='store_true',
        help="need segmentation"
    )
    parser.add_argument(
        '--vis', default=False, action='store_true',
        help="need to show visualization result"
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help="device ID (NPU ID) to use"
    )
    args = parser.parse_args()
    main(args)
