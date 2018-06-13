# implementation of proposal generation

This is a reimplementation of tag in PyTorch. 

For optical flow extraction and video list generation, you still need to use the original [TSN codebase](https://github.com/yjxiong/temporal-segment-networks).

## Training

To train a new model, use the `sgd_main_random.py` script.

The command to train the binary classifier can be 

```bash
python sgd_main.py ${dataset} ${modality} data/${dataset}_rgb_train_gt_list.txt data/${dataset}_rgb_val_gt_list.txt --arch BNInception --num_segments 5 --gd 20 --lr 0.005 --lr_steps ${step1} ${step2} --epochs 400 -b 128 -j 8 --dropout 0.7 --snapshot_pref ${dataset}_${modality}
```

For flow models:

```bash
python sgd_main.py ${dataset} ${modality} data/${dataset}_flow_train_gt_list.txt data/${dataset}_flow_val_gt_list.txt --arch BNInception --num_segments 5 --gd 20 --lr 0.005 --lr_steps ${step1} ${step2} --epochs 400 -b 128 -j 8 --dropout 0.7 --snapshot_pref ${dataset}_${modality}
```

## Testing

After training, there will checkpoints saved by pytorch, for example `*_checkpoint.pth`.

Use the following command to test its performance in the standard TSN testing protocol:

```bash
nohup python test_models.py activitynet1.2 data/activitynet1.2_tag_flow_val_proposal_list.txt ${model} --save_path ${save_dir1} > flow_test.log 2>&1 &

nohup python test_models.py activitynet1.2 data/activitynet1.2_tag_rgb_val_proposal_list.txt ${model} --save_path ${save_dir2} > rgb_test.log 2>&1 &

```

## fuse the action score

```bash
python fusion.py ${out_path} ${rgb_path} ${flow_path}
```

##### generate proposal with watershed
```bash
python watershed.py ${scores_dir} ${prop_file} --garma_left ${garma_left} --garma_right ${garma_right} --tao_left ${tao_left} --tao_right ${tao_right}
```

#### generate true proposals
```bash
python generate_true_props.py ${gt_file} ${prop_file} ${out_name}
```
