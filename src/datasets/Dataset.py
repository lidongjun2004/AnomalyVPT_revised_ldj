import os
import pickle
import random
import math
from collections import defaultdict

import loguru

from src.utils.tools import mkdir_if_missing, read_json, write_json, check_isfile

DATASET_SETTING = {
    'mvtec': {
        'clsnames': ['bottle', 'cable', 'capsule', 'carpet', 'hazelnut', 'leather', 'grid', 'pill',
                     'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood'],
        'dataset_dir': 'mvtec_ad',
        'image_dir': 'mvtec_anomaly_detection',
    },
    'visa': {
        'clsnames': [
            'candle', 'capsules', 'cashew', 'chewinggum',
            'fryum', 'macaroni1', 'macaroni2',
            'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
        ],
        'dataset_dir': 'VisA',
        'image_dir': 'visa/1cls',
    },
    'mpdd': {
        'clsnames': ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'],
        'dataset_dir': 'mpdd',
        'image_dir': 'MPDD',
    }
}


def build_dataset(cfg):
    return BaseDataset(name=cfg.DATASET.NAME,
                       root=cfg.DATASET.ROOT,
                       seed=cfg.SEED,
                       mix=cfg.DATASET.MIX,
                       num_shots=cfg.DATASET.NUM_SHOTS,
                       )


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        class_name (str): class name for labels. (e.g. 'normal', 'anomaly')
        mask_path (str): image mask path.
        specie_name (str): specification anomaly name.
        reality_name (str): the real class name of the item, which is label/class agnostic.
    """

    def __init__(self,
                 impath="",
                 label=0,
                 class_name="",
                 mask_path="",
                 specie_name="",
                 reality_name=""):
        assert isinstance(impath, str)
        assert check_isfile(impath)
        self.impath = impath
        self.label = label
        self.class_name = class_name
        self.mask_path = mask_path
        self.specie_name = specie_name
        self.reality_name = reality_name


class BaseDataset:
    """
    Args:
        name: the name of the dataset.
        root: the root of the dataset.
        seed: subsample seed.
        mix: if True, use the test set for generation tuning.
        num_shots: if <= 0, use full shots.
    """
    class_renames = ['perfect product', 'anomaly product']

    def __init__(self,
                 name='mvtec',
                 root='/data/users/ldj/dataset/',
                 seed=42,
                 mix=True,
                 num_shots=256):
        self.reality_names = DATASET_SETTING[name]['clsnames']
        self.test_reality_names = self.reality_names  # This attribute is prepared for testing specific classes.

        self.root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(self.root, DATASET_SETTING[name]['dataset_dir'])
        self.image_dir = os.path.join(self.dataset_dir, DATASET_SETTING[name]['image_dir'])
        self.meta_path = os.path.join(self.dataset_dir, 'meta.json')
        self.is_mix = mix
        if self.is_mix:
            self.split_path = os.path.join(self.dataset_dir, 'split_mix.json')
            self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_mix")
        else:
            self.split_path = os.path.join(self.dataset_dir, 'split.json')
            self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            meta_info = read_json(self.meta_path)
            # raw_trainval = meta_info['train']
            if self.is_mix:
                raw_trainval = meta_info['test']
            else:
                raw_trainval = meta_info['train']
            raw_test = meta_info['test']
            trainval = self.read_data(raw_trainval, self.class_renames, self.image_dir)
            train, val = self.split_trainval(trainval)
            test = self.read_data(raw_test, self.class_renames, self.image_dir)
            self.save_split(train, val, test, filepath=self.split_path, path_prefix=self.image_dir)

        if num_shots > 0:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(preprocessed):
                loguru.logger.info(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                loguru.logger.info(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        train, val, test = self.subsample_classes(train, val, test)
        self.train = self.select(train, reality_names=self.reality_names)
        self.val = self.select(val, reality_names=self.test_reality_names)
        self.test = self.select(test, reality_names=self.test_reality_names)

    @staticmethod
    def select(data_sources, reality_names):
        output = []
        for item in data_sources:
            if item.reality_name in reality_names:
                output.append(item)
        return output

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1):
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_reality(data_source)
            dataset = []

            for reality, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    sampled_items = items
                random.shuffle(sampled_items)
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    @staticmethod
    def split_dataset_by_reality(data_source):
        output = defaultdict(list)

        for item in data_source:
            output[item.reality_name].append(item)

        return output

    @staticmethod
    def read_data(meta_info, class_renames, image_dir):
        """
        meta_info = meta['train'] / meta['val']
        meta_info = {
            "cls_name": [
                "img_path":"",
                "cls_name":
                ...
            ]
        }
        aditems = [
            [
                impath,
                label,
                cls_name,
            ]
            ...
        ]
        """
        items = []
        reality_names = list(meta_info.keys())
        for reality_name in reality_names:
            lines = meta_info[reality_name]
            for line in lines:
                impath = os.path.join(image_dir, line['img_path'])
                if line['mask_path'].__len__() == 0 or line['mask_path'] is None:
                    mask_path = line['mask_path']
                else:
                    mask_path = os.path.join(image_dir, line['mask_path'])
                class_name = class_renames[line['anomaly']]
                specie_name = line['specie_name']
                label = line['anomaly']
                item = Datum(impath=impath, label=label, class_name=class_name,
                             mask_path=mask_path, specie_name=specie_name, reality_name=reality_name)
                items.append(item)
        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.):
        p_trn = 1 - p_val
        loguru.logger.info(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            reality_name = item.reality_name
            tracker[reality_name].append(idx)

        train, val = [], []
        for reality, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                class_name = item.class_name
                impath = impath.replace(path_prefix, "")

                mask_path = item.mask_path
                mask_path = mask_path.replace(path_prefix, "")
                specie_name = item.specie_name
                reality_name = item.reality_name
                if impath.startswith("/"):
                    impath = impath[1:]
                if mask_path.startswith("/"):
                    mask_path = mask_path[1:]
                out.append((impath, label, class_name, mask_path, specie_name, reality_name))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        loguru.logger.info(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, class_name, mask_path, specie_name, reality_name in items:
                impath = os.path.join(path_prefix, impath)
                if mask_path.__len__() > 0:
                    mask_path = os.path.join(path_prefix, mask_path)
                item = Datum(impath=impath, label=int(label), class_name=class_name,
                             mask_path=mask_path,
                             specie_name=specie_name,
                             reality_name=reality_name)
                out.append(item)
            return out

        loguru.logger.info(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        loguru.logger.info(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    class_name=item.class_name,
                    mask_path=item.mask_path,
                    reality_name=item.reality_name,
                    specie_name=item.specie_name
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output


if __name__ == '__main__':
    # test base dataset
    mvtec = BaseDataset(name='mvtec')
