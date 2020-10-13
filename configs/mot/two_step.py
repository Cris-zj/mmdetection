data = dict(
    data_root='data/MOT/MOT17/train/',
    img_prefix='img1',
    gt_file='gt/gt.txt',
    det_file='det/det.txt'
)

detection = dict(
    config=None,  # detection config
    checkpoint=None,  # detection model
    labels=[2, 4],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(512, 512),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=False),
                dict(type='Normalize',
                     mean=[0.0, 0.0, 0.0],
                     std=[255.0, 255.0, 255.0],
                     to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ]
        )
    ],
)

reid = dict(
    config=None,  # reid config
    checkpoint=None,  # reid model
    input_size=(256, 128),
    pipeline=[
        dict(type='Resize', size=(256, 128)),
        dict(type='Normalize',
             mean=[123.675, 116.28, 103.53],
             std=[58.395, 57.12, 57.375],
             to_rgb=True),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
    ]
)

tracktor = dict(
    metric=dict(
        type='cosine',
        match_thresh=0.2,
        budget=100,
    ),
    detection=dict(
        default=True,
        min_conf=0.8,
        min_height=0,
        nms_iou_thr=1.,
    )
)

evaluate = False
