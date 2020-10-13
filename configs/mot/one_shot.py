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
detection = dict(
    min_conf=0.5,
    min_height=0,
    nms_iou_thr=0.4,
)
mot = dict(
    similarity_metric='euclidean',
    similarity_thr=0.7,
    iou_thr=0.6,
    budget_size=30
)
evaluation = dict(
    iou_thr=0.5
)
