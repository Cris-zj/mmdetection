_base_ = [
    '_base_.py'
]

task_type = 'two_step'
model = dict(
    detector=dict(
        config='/home/zhujiao/AiriaCVLib/configs/test_pipeline_configs/resnet18_spp_pan_v4_crowdhuman_fall_modulewise.py',  # noqa: E501
        checkpoint='/home/zhujiao/AiriaCVLib/work_dirs/resnet18_spp_pan_v4_crowdhuman_fall_vf_608_30x_4lr_modulewise/epoch_300.pth',  # noqa: E501
    ),
    embeddor=dict(
        config='/home/zhujiao/repositories/mmclassification/configs/market1501/mobilenet_v2_relu_sls_patch_batch64.py',  # noqa: E501
        checkpoint='/home/zhujiao/repositories/mmclassification/work_dirs/market1501_mobilenetv2_relu_sls_patch_batch64/epoch_250.pth',  # noqa: E501
        data_cfg=dict(
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
    )
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
            img_scale=(512, 512),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=False),
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
