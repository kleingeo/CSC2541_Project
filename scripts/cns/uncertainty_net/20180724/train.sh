# CT intensities scaled from [-2000, +3000]

python train.py \
    --dataroot ./datasets/cns_2d_v02 \
    --name reg_uncertainty_net_20180724 \
    --model reg_unc_net \
    --which_model_netG unet_uncertainty_256 \
    --which_model_netD basic \
    --ngf 64 \
    --ndf 64 \
    --which_direction AtoB \
    --lambda_A 1 \
    --dataset_mode aligned2dnpy \
    --no_lsgan \
    --norm batch \
    --pool_size 0 \
    --input_nc 3 \
    --output_nc 1 \
    --save_epoch_freq 10 \
    --save_latest_freq 5000 \
    --display_port 8098 \
    --display_freq 500 \
    --print_freq 500 \
    --gpu_ids 0 \
    --batchSize 1 \
    --niter 25 \
    --niter_decay 25 \
    --init_type normal \
    --display_ncols 3
