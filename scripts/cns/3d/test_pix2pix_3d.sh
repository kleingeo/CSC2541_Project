python test.py \
    --dataroot ./datasets/cns \
    --name pix2pix_cns_3d_volume \
    --model pix2pix \
    --which_model_netG unet3d_256 \
    --which_direction AtoB \
    --dataset_mode aligned3d \
    --norm batch3d \
    --input_nc 3 \
    --output_nc 1 \
    --ngf 64 \
    --ndf 64 \
    --display_port 8098 \
    --gpu_ids 1 \
    --how_many 100 \
    --phase test
