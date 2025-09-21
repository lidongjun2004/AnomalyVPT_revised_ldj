import os
import sys
import argparse
import torch

# 设置环境变量（模拟train_gpu.sh的行为）
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 模拟命令行参数（基于train_gpu.sh的配置）
class DebugArgs:
    def __init__(self):
        self.config_file = "./configs/vitl14_ep20.yaml"
        self.output_dir = "./output/train_vpt_mvtec_-2_seg_cls_1024_output_dim_gs05_3/"
        self.name = "mvtec"
        self.seed = 1003
        self.device = 0
        self.pixel = True
        self.eval = False
        self.vis = False
        self.resume = ""

# 创建模拟参数对象
args = DebugArgs()

# 导入主程序模块
from main import main, setup_cfg, reset_cfg, setup_logger
from src.utils.tools import set_random_seed
from configs.default import _C as cfg_default

# 覆盖main.py中的main函数以支持调试
def debug_main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print(f"Setting fixed seed: {cfg.SEED}")
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    device = torch.device(f"cuda:{cfg.DEVICE}" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu")
    print(f"Using device: {device}")

    # 在这里可以设置断点查看模型结构
    if cfg.EVAL:
        from src.test import test
        test(cfg, device=device)
    else:
        from src.train import train
        train(cfg, device=device)

# 启动调试
if __name__ == "__main__":
    # 设置随机种子以便复现
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print("="*50)
    print("Starting debug session")
    print(f"Config: {args.config_file}")
    print(f"Output dir: {args.output_dir}")
    print("="*50)
    
    # 调用主函数
    debug_main(args)