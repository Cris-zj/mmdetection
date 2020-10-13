import os
import numpy as np
from torch.utils.data import Dataset

from mmdet.datasets.pipelines import Compose


class Sequence(Dataset):

    def __init__(self,
                 img_prefix,
                 ann_file,
                 pipeline,
                 seqinfo_file=None):
        self.img_prefix = img_prefix
        self.ann_file = ann_file
        self.samples = make_dataset(img_prefix, ann_file)
        self.data_infos = self.load_annotations()
        self.pipeline = Compose(pipeline)
        self.seqinfo = parse_seqinfo(seqinfo_file)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data = self.data_infos[idx]
        return self.pipeline(data)

    def load_annotations(self):
        data_infos = []
        for filename, gt_ids, gt_bboxes, gt_labels in self.samples:
            info = {'img_prefix': self.img_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_ids'] = gt_ids
            info['gt_bboxes'] = gt_bboxes
            info['gt_labels'] = gt_labels
            data_infos.append(info)
        return data_infos


def make_dataset(img_dir, gt_file):
    image_filenames = {
        int(os.path.splitext(f)[0]): f
        for f in os.listdir(img_dir)}

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        raise RuntimeError('Image data is None!')

    groundtruth = np.loadtxt(gt_file, delimiter=',')

    data = []
    for frame_idx in range(min_frame_idx, max_frame_idx + 1):
        img = image_filenames[frame_idx]
        gt = groundtruth[groundtruth[:, 0] == frame_idx]
        gt_ids = gt[:, 1]
        gt_bboxes = gt[:, 2:6]
        gt_labels = gt[:, 7]
        data.append((img, gt_ids, gt_bboxes, gt_labels))

    return data


def parse_seqinfo(info_file):
    if os.path.exists(info_file):
        with open(info_file, "r") as f:
            line_splits = [
                line.split('=') for line in f.read().splitlines()[1:]
            ]
            info_dict = dict()
            for s in line_splits:
                if isinstance(s, list) and len(s) == 2:
                    info_dict[s[0].lower()] = int(
                        s[1]) if s[1].isdigit() else s[1]
        return info_dict
    else:
        return None


def build_dataset(img_prefix,
                  ann_file,
                  pipeline,
                  seqinfo_file=None):
    dataset = Sequence(img_prefix,
                       ann_file,
                       pipeline,
                       seqinfo_file=seqinfo_file)
    return dataset
