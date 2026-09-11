from src.functs import read_image,  save_marker, gen_svox_center
import os
import numpy as np
import pandas as pd
from CONFIG import *
from src.flim_tools import load_patches_from_marked_images


def save_patches_labels(res_dir, labels):

    np.save(f"{res_dir}/labels.npy", labels)


def save_patches_layer(res_dir, patches, layer):
    np.save(f"{res_dir}/not_norm_patches_l{layer}.npy", patches)

def compute_receptive_field(layer, strides, k_size):
    rf = 1
    jump = 1
    for l in range(0,layer):
        rf += (k_size-1)*jump
        jump *=strides[l]

    return rf

def gen_patches(dataset, split, n_svox, k_size, strides):


    res_dir = f"s_patches/{dataset}/s_{split}/n_{n_svox}/k_{k_size}"

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)


    config_d  = config[dataset]
    GT_DIR = config_d["label_dir"]
    IM_DIR = config_d["orig_dir"]
    

    use_lab = False
    if config_d["color"]:
        use_lab=True

    marker_dir = f"s_markers/{dataset}/s_{split}/n_{n_svox}"

    for layer in range(0, len(strides)):
        rf = compute_receptive_field(layer+1, strides, k_size)
        kernel_size = [rf,rf]
        patches, labels, ilabel = load_patches_from_marked_images(marker_dir, IM_DIR, GT_DIR, kernel_size=kernel_size, use_lab=use_lab)

        if layer == 0:
            save_patches_labels(res_dir, labels)
        save_patches_layer(res_dir, patches, layer+1)





if __name__ == "__main__":

    split = 3
    strides = [1,2,1]

    for n_svox in [25]:#50
        for kernel_size in [3,5,9]:

            for dataset in ['plants','schisto','refuge', 'fish', 'gbm2d', 'mass_building']:
                print(f"Getting patches from {dataset} {split} {n_svox} {kernel_size}")
                gen_patches(dataset, split, n_svox, kernel_size, strides)