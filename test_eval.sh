# this script is used to test destr
nproc_per_node=8
batch_size=2

# nproc_per_node=4
# batch_size=4


backbone=resnet50
coco_path=../../data/coco/
output_dir=../useful_weights_20220904/destr6_r50_d256h8_20220916_epoch49_4360/
epochs=50
resume_weights=${output_dir}/checkpoint0049.python
lr=1e-4
hidden_dim=256

python -m torch.distributed.launch --nproc_per_node ${nproc_per_node} \
        --use_env main.py               \
        --eval                          \
        --coco_path ${coco_path}        \
        --output_dir ${output_dir}      \
        --epochs ${epochs}              \
        --hidden_dim ${hidden_dim}      \
        --lr ${lr}                      \
        --batch_size ${batch_size}      \
        --resume  ${resume_weights}