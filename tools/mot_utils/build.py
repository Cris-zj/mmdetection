import torch
import numpy as np

import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter, MMDataParallel

from mmcls.datasets.pipelines import Compose as reid_Compose
from mmcls.models import build_reid
from mmdet.models import build_tracktor, build_detector


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def extract_image_patch(image, bbox):
    bbox = bbox.astype(np.int)
    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    return image


def build_tracktor_encoder(config, checkpoint, device='cuda:0'):
    print('Building joint detector and embedding ...')
    model_cfg = mmcv.Config.fromfile(config)
    model_cfg.model.pretrained = None
    model = build_tracktor(model_cfg.model, test_cfg=model_cfg.test_cfg)
    _ = load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    def encoder(data):
        with torch.no_grad():
            return model(return_loss=False, rescale=True, **data)

    return encoder


def build_detect_encoder(config, checkpoint, device='cuda:0'):
    print('Building detector ...')
    model_cfg = mmcv.Config.fromfile(config)
    model_cfg.model.pretrained = None
    model = build_detector(model_cfg.model, test_cfg=model_cfg.test_cfg)
    _ = load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    def encoder(data):
        with torch.no_grad():
            return model(return_loss=False, rescale=True, **data)

    return encoder


def build_embed_encoder(config, checkpoint, data_cfg, device='cuda:0'):
    # build feature embeddor for reid
    print('Building feature embeddor ...')
    embed_cfg = mmcv.Config.fromfile(config)
    embed_cfg.model.pretrained = None
    embeddor = build_reid(embed_cfg.model)
    _ = load_checkpoint(embeddor, checkpoint, map_location='cpu')
    embeddor.to(device)
    embeddor.eval()
    embeddor.forward = embeddor.extract_feat

    input_size = data_cfg.input_size
    patch_pipeline = [LoadImage()] + data_cfg.pipeline
    patch_transform = reid_Compose(patch_pipeline)

    def embed_encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box)
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., input_size).astype(np.uint8)
            data = patch_transform(dict(img=patch))
            image_patches.append(data)
        if len(image_patches) == 0:
            return torch.zeros((0, 512)).to(device)
        data = scatter(collate(
            image_patches, samples_per_gpu=len(image_patches)), [device])[0]
        with torch.no_grad():
            return embeddor(**data)

    return embed_encoder
