dataset_type = 'CocoDataset'
classes = ('a', )
custom_imports = dict(
    imports=[
        'Algorithms.FISH_deployment.Swim_Fish.datasets.pipelines', 'Algorithms.FISH_deployment.Swim_Fish.datasets.FishDataset',
        'Algorithms.FISH_deployment.Swim_Fish.models'
    ],
    allow_failed_imports=False)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='FishDataset',
        classes=('a', ),
        ann_file=
        '/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Annotations/instances_all_0929_her2_dense.json',
        img_prefix=
        '/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Her2_Images_40X_szl_dense/',
        pipeline=[
            dict(type='LoadImageSignalFromFile'),
            dict(
                type='LoadFishAnnotations',
                with_bbox=True,
                with_mask=True,
                with_seg=True,
                with_signal=True,
                with_ignore_area=False),
            dict(type='Resize_Fish', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip_Fish', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='SegRescale', scale_factor=0.125),
            dict(type='DefaultFormatBundle_Fish'),
            dict(
                type='Collect_Fish',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_masks',
                    'gt_signal_points', 'gt_signal_labels', 'gt_semantic_seg'
                ])
        ],
        seg_prefix=
        '/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Get_Stuffthing/stuffthingmaps_Her2_0929_dense/'
    ),
    val=dict(
        type='FishDataset',
        classes=('a', ),
        ann_file=
        '/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Annotations/instances_all_0929_her2_dense.json',
        img_prefix=
        '/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Her2_Images_40X_szl_dense/',
        pipeline=[
            dict(type='LoadImageSignalFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize_Fish', keep_ratio=True),
                    dict(type='RandomFlip_Fish', flip_ratio=0.5),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect_Fish', keys=['img'])
                ])
        ]),
    test=dict(
        type='FishDataset',
        classes=('a', ),
        ann_file=
        '/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Annotations/instances_all_0929_her2_dense.json',
        img_prefix=
        '/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Her2_Images_40X_szl_dense/',
        pipeline=[
            dict(type='LoadImageSignalFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize_Fish', keep_ratio=True),
                    dict(type='RandomFlip_Fish', flip_ratio=0.5),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect_Fish', keys=['img'])
                ])
        ]))
evaluation = dict(metric=['bbox', 'segm', 'signal'])
optimizer = dict(
    type='SGD',
    lr=0.06,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            classification_branch=dict(lr_mult=0.2, decay_mult=1.0),
            regression_branch=dict(lr_mult=0.2, decay_mult=1.0),
            conv_delta=dict(lr_mult=0.2, decay_mult=1.0),
            conv_logits=dict(lr_mult=0.2, decay_mult=1.0))))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=50)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
mmdet_base = '../../thirdparty/mmdetection/configs/_base_'
model = dict(
    type='HybridTaskSignalCascade',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='HTCFishRoIUNetHead',
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ],
        signal_roi_extractor=[
            dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[16]),
            dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8])
        ],
        signal_head=[
            dict(
                type='HTCSignalUNetHead',
                return_logit=False,
                num_convs=4,
                roi_feat_size=14,
                in_channels=256,
                conv_kernel_size=3,
                conv_out_channels=128,
                num_classes=1,
                signal_classes=2,
                loss_points_coef=dict(type='MSESignalLoss', loss_weight=0.002),
                loss_cls_coef=dict(
                    type='CrossEntropySignalLoss', loss_weight=1),
                signal_loss_weight=1.0,
                eos_coef=0.1,
                red_coef=0.8),
            dict(
                type='HTCSignalUNetHead',
                return_logit=True,
                num_convs=4,
                roi_feat_size=14,
                in_channels=384,
                conv_kernel_size=3,
                conv_out_channels=64,
                num_classes=1,
                signal_classes=2,
                loss_points_coef=dict(type='MSESignalLoss', loss_weight=0.002),
                loss_cls_coef=dict(
                    type='CrossEntropySignalLoss', loss_weight=1),
                signal_loss_weight=1.0,
                eos_coef=0.1,
                red_coef=0.8)
        ],
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=183,
            loss_seg=dict(
                type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5,
            signal_thr_binary=0.5)))
work_dir = './work_dirs/1017_dense_szl_lr0.06_size14_mul_0.2_coef0.1_red0.8'
cfg_name = '1017_dense_szl_lr0.06_size14_mul_0.2_coef0.1_red0.8'
auto_resume = False
gpu_ids = [3]
