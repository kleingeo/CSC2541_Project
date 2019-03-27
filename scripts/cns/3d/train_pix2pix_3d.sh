python train.py \
    --dataroot ./datasets/cns \
    --name pix2pix_cns_3d_volume02 \
    --model pix2pix \
    --which_model_netG unet3d_256 \
    --which_model_netD basic3d \
    --ngf 64 \
    --ndf 64 \
    --which_direction AtoB \
    --lambda_A 100 \
    --dataset_mode aligned3d \
    --no_lsgan \
    --norm batch3d \
    --pool_size 0 \
    --input_nc 3 \
    --output_nc 1 \
    --save_epoch_freq 10 \
    --save_latest_freq 500 \
    --display_port 8098 \
    --display_freq 10 \
    --print_freq 10 \
    --gpu_ids 0 \
    --batchSize 1 \
    --niter 100 \
    --niter_decay 100 \
    --init_type normal \
    --display_ncols 3
