root=../GGAE
data=$root/data
output_dir=$root/checkpoints/fb/out/

CUDA_VISIBLE_DEVICES=1 python $root/main.py \
    --data=$data/FB15k-237/ \
    --output_folder=$output_dir \
    --batch_size_gat=272115 \
    --weight_decay_gat=0.00001 \
    --epochs_gat=3000 \
    --margin=1 \
    --epochs_conv=200 \
    --valid_invalid_ratio_conv=40 \
    --drop_conv=0.3 \
    --out_channels=50 \
    --no_2hop \
    --no_reverse2hop \
    --n_path=2 \
