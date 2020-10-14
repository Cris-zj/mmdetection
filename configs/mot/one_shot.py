_base_ = [
    '_base_.py'
]

task_type = 'one_shot'
model = dict(
    tracktor=dict(
        config='configs/personsearch/yolo_embedding.py',
        checkpoint='pretrained/jde.uncertainty.pth')
)

data_root = 'data/MOT/MOT17/train/'
data = dict(
    seq_prefix=data_root,
    img_prefix='img1/',
    ann_file='gt/gt.txt',
    seqinfo_file='seqinfo.ini',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1088, 608),
            flip=False,
            transforms=[
                dict(type='LetterBox',
                     size=(1088, 608),
                     border_value=(127.5, 127.5, 127.5),
                     interpolation='area'),
                dict(type='Normalize',
                     mean=[0.0, 0.0, 0.0],
                     std=[255.0, 255.0, 255.0],
                     to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=[
                     'img', 'gt_ids', 'gt_bboxes', 'gt_labels']),
            ]
        )
    ],
)

post_processing = dict(
    min_conf=0.5,
    min_height=0,
    nms_iou_thr=0.4,
)
