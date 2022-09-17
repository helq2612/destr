# this script is used to train destr on OSU hpc
nproc_per_node=8
batch_size=2

# nproc_per_node=4
# batch_size=4

backbone=resnet101
backbone=resnet50
coco_path=../../data/coco/
output_dir=destr6_d256h8_${backbone}_20220916
epochs=50
lr_drop=40
lr=1e-4
hidden_dim=256
num_queries=300
python -m torch.distributed.launch --nproc_per_node ${nproc_per_node} \
        --use_env main.py               \
        --coco_path ${coco_path}        \
        --output_dir ${output_dir}      \
        --epochs ${epochs}              \
        --backbone ${backbone}          \
        --hidden_dim ${hidden_dim}      \
        --lr ${lr}                      \
        --lr_drop ${lr_drop}            \
        --batch_size ${batch_size}      \
        --num_queries ${num_queries}    \

