dataset=$1
modality=$2
step1=$3
step2=$4

source /home/disk1/vis/lijun/set_python3.sh
#$### training classifier
nohup python sgd_main.py ${dataset} ${modality} data/${dataset}_rgb_train_gt_list.txt data/${dataset}_rgb_val_gt_list.txt --arch BNInception --num_segments 5 --gd 20 --lr 0.005 --lr_steps ${step1} ${step2} --epochs 400 -b 128 -j 8 --dropout 0.7 --snapshot_pref ${dataset}_${modality} > flow_activitynet1.2_train.log 2>&1 &

save_dir1=$5
save_dir2=$6
model=$7
###### predict scores
nohup python test_models.py activitynet1.2 data/activitynet1.2_tag_flow_val_proposal_list.txt ${model} --save_path ${save_dir1} > rgb_test.log 2>&1 &

nohup python test_models.py activitynet1.2 data/activitynet1.2_tag_rgb_val_proposal_list.txt ${model} --save_path ${save_dir2} > rgb_test.log 2>&1 &

############ fuse rgb and flow score
out_path=$8
rgb_path=$9
flow_path=$10

nohup python fusion.py ${out_path} ${rgb_path} ${flow_path} > aaa.log 2>&1 &


########## generate propsoals

garma_left=$11
garma_right=$12
tao_left=$13
tao_right=$14
scores_dir=$15
prop_file=$16

nohup python watershed.py ${scores_dir} ${prop_file} --garma_left ${garma_left} --garma_right ${garma_right} --tao_left ${tao_left} --tao_right ${tao_right} > prop3.log 2>&1 &


### generate ssn format proposals
gt_file=$17
prop_file=$18
outname=$19
nohup python generate_true_props.py ${gt_file} ${prop_file} ${out_name}
