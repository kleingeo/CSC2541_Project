#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:18:37 2019

@author: mhemsley
"""
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

directory = "/home/mhemsley/BRATS18/MICCAI_BraTS_2018_Data_Training/HGG/"

folder_list= os.listdir(directory)
x=0
i=0
while i < len(folder_list):
    path_t1 = os.path.join(directory+folder_list[i], folder_list[i]+'_t1.nii.gz')
    path_t2 = os.path.join(directory+folder_list[i], folder_list[i]+'_t2.nii.gz')
    path_t1ce = os.path.join(directory+folder_list[i], folder_list[i]+'_t1ce.nii.gz')
    path_flair = os.path.join(directory+folder_list[i], folder_list[i]+'_flair.nii.gz')
    path_seg = os.path.join(directory+folder_list[i], folder_list[i]+'_seg.nii.gz')
    
    vol_t1=nib.load(path_t1)
    vol_t2=nib.load(path_t2)
    vol_t1ce=nib.load(path_t1ce)
    vol_flair=nib.load(path_flair)
    vol_seg=nib.load(path_seg) 
    
    vol_t1_np=np.array(vol_t1.dataobj)
    vol_t2_np=np.array(vol_t2.dataobj)
    vol_t1ce_np=np.array(vol_t1ce.dataobj)
    vol_flair_np=np.array(vol_flair.dataobj)
    vol_seg_np=np.array(vol_seg.dataobj)
    vol_seg_np[vol_seg_np > 1.0] = 1.0
    
    t1_norm = (vol_t1_np - np.min(vol_t1_np))/np.ptp(vol_t1_np)
    t2_norm = (vol_t2_np - np.min(vol_t2_np))/np.ptp(vol_t2_np)
    t1ce_norm = (vol_t1ce_np - np.min(vol_t1ce_np))/np.ptp(vol_t1ce_np)
    flair_norm = (vol_flair_np - np.min(vol_flair_np))/np.ptp(vol_flair_np)
    seg_norm = (vol_seg_np - np.min(vol_seg_np))/np.ptp(vol_seg_np)
    
    ii=0
    while ii<t1_norm.shape[2]:
        t1_slice=t1_norm[:,:,ii]
        t2_slice=t2_norm[:,:,ii]
        t1ce_slice=t1ce_norm[:,:,ii]
        flair_slice=flair_norm[:,:,ii]
        seg_slice=seg_norm[:,:,ii]
        
       
        t1_slice=np.reshape(t1_slice,(1,240,240,1))
        t2_slice=np.reshape(t2_slice,(1,240,240,1))
        t1ce_slice=np.reshape(t1ce_slice,(1,240,240,1))
        flair_slice=np.reshape(flair_slice,(1,240,240,1))
        seg_slice=np.reshape(seg_slice,(1,240,240,1))
        
        full_slice=np.concatenate((t1ce_slice,t2_slice,flair_slice),axis=0).astype('float32')
           
        if i<=len(folder_list)*0.377:
            np.save('/home/mhemsley/Pix2PixCheckAngus/pytorch-CycleGAN-and-pix2pix/datasets/BRATS18Slices/trainA/'+folder_list[i]+'_'+str(i)+'_'+str(ii), t1_slice)
            np.save('/home/mhemsley/Pix2PixCheckAngus/pytorch-CycleGAN-and-pix2pix/datasets/BRATS18Slices/trainB/'+folder_list[i]+'_'+str(i)+'_'+str(ii), t2_slice)
            np.save('/home/mhemsley/Pix2PixCheckAngus/pytorch-CycleGAN-and-pix2pix/datasets/BRATS18Slices/trainB_flair/'+folder_list[i]+'_'+str(i)+'_'+str(ii), flair_slice)
            
            if ii==0:
                text_file = open("CS2541_Project_cGAN_training_list.txt","a")
                text_file.write(folder_list[i]+ "\n")
                text_file.close()
            
        elif i<=len(folder_list)*0.5:
            np.save('/home/mhemsley/Pix2PixCheckAngus/pytorch-CycleGAN-and-pix2pix/datasets/BRATS18Slices/testA/'+folder_list[i]+'_'+str(i)+'_'+str(ii), t1_slice)
            np.save('/home/mhemsley/Pix2PixCheckAngus/pytorch-CycleGAN-and-pix2pix/datasets/BRATS18Slices/testB/'+folder_list[i]+'_'+str(i)+'_'+str(ii), t2_slice)
            np.save('/home/mhemsley/Pix2PixCheckAngus/pytorch-CycleGAN-and-pix2pix/datasets/BRATS18Slices/testB_flair/'+folder_list[i]+'_'+str(i)+'_'+str(ii), flair_slice)
            
            if ii==0:
                text_file = open("CS2541_Project_cGAN_testing_list.txt","a")
                text_file.write(folder_list[i]+ "\n")
                text_file.close()
            
        elif i<=len(folder_list)*0.877:
            np.save('/home/mhemsley/Pix2PixCheckAngus/pytorch-CycleGAN-and-pix2pix/datasets/BRATS18Slices/seg_trainA/'+folder_list[i]+'_'+str(i)+'_'+str(ii), t1_slice)
            np.save('/home/mhemsley/Pix2PixCheckAngus/pytorch-CycleGAN-and-pix2pix/datasets/BRATS18Slices/seg_trainB/'+folder_list[i]+'_'+str(i)+'_'+str(ii), t2_slice)
            np.save('/home/mhemsley/Pix2PixCheckAngus/pytorch-CycleGAN-and-pix2pix/datasets/BRATS18Slices/seg_trainB_flair/'+folder_list[i]+'_'+str(i)+'_'+str(ii), flair_slice)
            
            if ii==0:
                text_file = open("CS2541_Project_seg_training_list.txt","a")
                text_file.write(folder_list[i]+ "\n")
                text_file.close()
            
        else:
            np.save('/home/mhemsley/Pix2PixCheckAngus/pytorch-CycleGAN-and-pix2pix/datasets/BRATS18Slices/seg_testA/'+folder_list[i]+'_'+str(i)+'_'+str(ii), t1_slice)
            np.save('/home/mhemsley/Pix2PixCheckAngus/pytorch-CycleGAN-and-pix2pix/datasets/BRATS18Slices/seg_testB/'+folder_list[i]+'_'+str(i)+'_'+str(ii), t2_slice)
            np.save('/home/mhemsley/Pix2PixCheckAngus/pytorch-CycleGAN-and-pix2pix/datasets/BRATS18Slices/seg_testB_flair/'+folder_list[i]+'_'+str(i)+'_'+str(ii), flair_slice)
            
            if ii==0:
                text_file = open("CS2541_Project_seg_testing_list.txt","a")
                text_file.write(folder_list[i]+ "\n")
                text_file.close()
        
        ii=ii+1
        
    i=i+1
          






