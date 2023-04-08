## DataSet Prepare

make the image & annotations follow below format
```
--OpenLaneV2
    |-- av2_map_anns_val.json  # used for val record the gt annotations
    |-- test
    |-- test_annotations.json
    |-- train
    |-- train_annotations.json
    |-- val
    `-- val_annotations.json
```

## Train
```bash
cd CVPR_HDMAP;
zsh tools/dist_train.sh projects/configs/custom/maptr_tiny_r50_24e_800x1024 ngpus
```

## Vis
```bash
cd CVPR_HDMAP;
export PYTHONPATH=CVPR_HDMAP
python tools/maptr/vis_pred_av2.py config_path ckpt_path
```

## Note
- Image proces: Src image resize to (1550x2048) ---> scale(0.5) ---->  775x1024 ---> padding ---> 800x1024
- 由于盛银使用集群读取图片，因此其他人在使用时需要将 `ann_file_s3` 注释掉



## 模型

| Config | Ped_crossing | Divider | Boundary| mAP| | 
| :---: | :---: | :---: | :---: | :---:|:---:|
| maptr_tiny_r50_24e_800x1024_seg | 50.79 | 61.94 | 60.20 | 57.64

