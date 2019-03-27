python test.py \
    --dataroot ./datasets/cns_2d_v02 \
    --name pix2pix_2d_cns_20180607 \
    --model pix2pix \
    --which_model_netG unet_256 \
    --which_direction AtoB \
    --dataset_mode aligned2dnpy \
    --norm batch \
    --input_nc 3 \
    --output_nc 1 \
    --ngf 64 \
    --ndf 64 \
    --display_port 8098 \
    --gpu_ids 1 \
    --how_many 2000
