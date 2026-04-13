# dataset settings
dataset_type = 'MydataDataset'
data_root = '/opt/data/private/Syg/Semantic/Dataset/RSSOD/EORSSD'
img_norm_cfg = dict(
    mean=[89.35, 96.94, 91.85], std=[59.27, 50.63, 46.23], to_rgb=True) #train cod mean[115.97,112.23,87.02] std[66.32,64.15,67.30]
img_norm_cfg_test = dict(
    mean=[93.46, 99.73, 94.31], std=[56.12, 46.93, 44.24], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='SETR_Resize', keep_ratio=True, crop_size=(512, 512), setr_multi_scale=True),
            #dict(type='Resize', keep_ratio=True),
            #dict(type='RandomFlip'),
            #dict(type='Normalize', **img_norm_cfg_test),
            #dict(type='ImageToTensor', keys=['img']),
            #dict(type='Collect', keys=['img']),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip', prob=1.0),
            dict(type='Normalize', **img_norm_cfg_test),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
"""
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='SETR_Resize', keep_ratio=True, crop_size=(512, 512), setr_multi_scale=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip', prob=1.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
"""
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))
