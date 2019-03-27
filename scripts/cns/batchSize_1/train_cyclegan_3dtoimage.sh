python train.py \
    --dataroot ./datasets/cns \
    --name cyclegan_3d_cns \
    --model cycle_gan \
    --dataset_mode unaligned3dtoimage \
    --pool_size 50 \
    --display_port 8098 \
    --input_nc 1 \
    --output_nc 1 \
    --which_direction AtoB \
    --display_freq 200 \
    --print_freq 100 \
    --niter 100 \
    --niter_decay 100 \
    --init_type normal \
    --gpu_ids 0 \
    --batchSize 1
#    --no_dropout \
