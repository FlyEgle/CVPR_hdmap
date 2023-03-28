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
zsh tools/dist_train.sh projects/configs/maptr/maptr_tiny_r50_24e_1gpus_av2_resize_intrinsic.py ngpus
```

## Vis
```bash
cd CVPR_HDMAP;
export PYTHONPATH=CVPR_HDMAP
python tools/maptr/vis_pred_av2.py config_path ckpt_path
```

## Note
- Image proces: Src image resize to (1600, 900)
