import argparse
import os
import json

import loguru

CLSNAMES = {
    'mvtec': [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    ],
    'visa': [
        'candle', 'capsules', 'cashew', 'chewinggum',
        'fryum', 'macaroni1', 'macaroni2',
        'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
    ],
    'mpdd': [
        'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'
    ]
}


class MetaSolver(object):
    def __init__(self, root='data/mvtec', name='mvtec'):
        assert name in CLSNAMES.keys(), f"Name {name} is not in the supported dataset names: {CLSNAMES.keys()}"
        self.root = root
        self.clsnames = CLSNAMES[name]
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.clsnames:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples = anomaly_samples + 1
                            else:
                                normal_samples = normal_samples + 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        loguru.logger.info('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)


# if __name__ == '__main__':
#     # runner = MetaSolver(root='/hy-tmp/ccy/datasets/VisA/visa/1cls', name='visa')
#     # runner = MetaSolver(root='/data1/ccy/datasets/mvtec_ad/mvtec_anomaly_detection', name='mvtec')
#     # runner = MetaSolver(root='/data1/ccy/datasets/mpdd/MPDD', name='mpdd')
#     runner.run()


def main():
    parser = argparse.ArgumentParser(description='Run MetaSolver with specified dataset.')

    parser.add_argument('--root', type=str, required=True,
                        default='/data1/ccy/datasets/mvtec_ad/mvtec_anomaly_detection',
                        help='Root directory of the dataset.')
    parser.add_argument('--name', type=str, required=True,
                        default='mvtec',
                        help='Name of the dataset.')

    args = parser.parse_args()

    runner = MetaSolver(root=args.root, name=args.name)
    runner.run()


if __name__ == '__main__':
    main()
