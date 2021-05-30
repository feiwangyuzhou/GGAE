root=../GGAE
data=$root/data
output_dir=$root/checkpoints/kinship/out

CUDA_VISIBLE_DEVICES=0 python $root/main.py \
    --data=$data/kinship/ \
    --output_folder=$output_dir \
    --batch_size_gat=8544 \
    --weight_decay_gat=0.000005 \
    --epochs_gat=3600 \
    --margin=5 \
    --epochs_conv=200 \
    --valid_invalid_ratio_conv=10 \
    --drop_conv=0.0 \
    --out_channels=500 \
    --no_2hop \
    --no_reverse2hop \
    --n_path=10 \
