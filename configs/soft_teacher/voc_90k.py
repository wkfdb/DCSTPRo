_base_ = "base.py"


classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

model = dict(
    roi_head=dict(
        type="StandardRoIHead2",
        bbox_head = dict(
            type='Shared2FCBBoxHead2',
            num_classes=20,
            loss_cls=dict(
                type='mySoftCrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0
            ),
        )
    ),
)


semi_wrapper = dict(
    type="DCSTPro",
    train_cfg=dict(
        rpn_pseudo_threshold=0.7,
        cls_pseudo_threshold=0.7,
        reg_pseudo_threshold=0.7,
    )
)


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    step=[60000, 80000])
optimizer = dict(type="SGD", lr=0.015, momentum=0.9, weight_decay=0.0001)
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=90000)


data = dict(
    samples_per_gpu=10,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            type="VocDataset",
            ann_file="data/voc/annotations/voc07_trainval.json",
            img_prefix="data/voc/",
            classes=classes,
        ),
        unsup=dict(
            type="VocDataset",
            ann_file="data/voc/annotations/voc12_trainval.json",
            img_prefix="data/voc/",
            classes=classes,
        ),
    ),
    val=dict(
        type="VocDataset",
        ann_file="data/voc/annotations/voc07_test.json",
        img_prefix="data/voc/", 
        classes=classes,
    ),
    test=dict(
        type="VocDataset",
        ann_file="data/voc/annotations/voc07_test.json",
        img_prefix="data/voc/",
        classes=classes,
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

fold = 1
percent = 1

work_dir = "work_dirs/voc"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
