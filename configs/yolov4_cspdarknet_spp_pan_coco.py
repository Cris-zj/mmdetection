model = dict(
    type='SingleStageDetector',
    pretrained='pretrained/cspdarknet53_omega_pretrained.pth',
    backbone=dict(
        type='DarkNet',
        depth=53,
        num_stages=5,
        with_csp=True,
        out_indices=(2, 3, 4),
        conv_cfg=None,
        norm_cfg=dict(type='BN', eps=1e-4, momentum=0.03),
        norm_eval=False,
        activation_cfg=dict(type='Mish')
    ),
    neck=[
        dict(
            type='SPP',
            in_channels=1024,
            out_channels=[512, 1024, 512],
            kernel_size=[1, 3, 1],
            stride=[1, 1, 1],
            padding=[0, 1, 0],
            out_pool_size=[5, 9, 13],
            conv_cfg=None,
            norm_cfg=dict(type='BN', eps=1e-4, momentum=0.03),
            activation_cfg=dict(
                type='LeakyReLU', negative_slope=0.1, inplace=True)),
        dict(
            type='PANYOLO',
            num_levels=3,
            in_channels=[256, 512, 2048],
            out_channels=[128, 256, 512],
            num_blocks=[2, 2, 1],
            extra_convs_on_inputs=True,
            conv_cfg=None,
            norm_cfg=dict(type='BN', eps=1e-4, momentum=0.03),
            activation_cfg=dict(
                type='LeakyReLU', negative_slope=0.1, inplace=True))
        ],
    bbox_head=dict(
        type='YOLOV4Head',
        num_classes=81,
        scale_x_y=[1.2, 1.1, 1.05],
        in_channels=[128, 256, 512],
        anchors=[([12, 16], [19, 36], [40, 28]),
                 ([36, 75], [76, 55], [72, 146]),
                 ([142, 110], [192, 243], [459, 401])],
        anchor_strides=[8, 16, 32],
        conv_cfg=None,
        norm_cfg=dict(type='BN', eps=1e-4, momentum=0.03),
        activation_cfg=dict(
            type='LeakyReLU', negative_slope=0.1, inplace=True),
        loss_bbox=dict(
            type='CompleteIoULoss',
            reduction='mean',
            loss_weight=1.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0)
    )
)
train_cfg = dict(
    iou_thresh=0.213,
    ignore_thresh=0.7,
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.6),
    min_bbox_size=0,
    score_thr=0.001,
    max_per_img=100)
dataset_type = 'CocoDataset'
data_root = 'data/coco2017/'
img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='JitterCrop', jitter_ratio=0.3),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomDistort', hue=0.1, saturation=1.5, exposure=1.5),
    dict(type='Mosaic',
         min_offset_ratio=0.2,
         total_num_mosaic_images=4),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        test_mode=True)
)
# optimizer
optimizer = dict(type='SGD', lr=0.00261, momentum=0.949, weight_decay=5e-4,
                 paramwise_options=dict(
                     bias_lr_mult=2.,
                     bias_decay_mult=0.,
                     norm_decay_mult=0.),
                 auto_adjust=True, base_batch_size=64)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='pow',
    warmup_iters=1000,
    warmup_ratio=1.0 / 4,
    step=[218, 246, 273, 291])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 300
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/yolov4_cspdarknet53_spp_pan_coco'
load_from = None
resume_from = None
workflow = [('train', 1)]
