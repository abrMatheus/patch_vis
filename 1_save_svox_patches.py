from src.functs import read_image,  save_marker, gen_svox_center
import os
import numpy as np
import pandas as pd
from CONFIG import *
from src.flim_tools import load_patches_from_marked_images


def save_patches(res_dir, patches, labels):

    np.save(f"{res_dir}/not_norm_patches.npy", patches)
    np.save(f"{res_dir}/labels.npy", labels)


def gen_patches(dataset, split, n_svox, k_size):


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

    kernel_size = [k_size,k_size]
    patches, labels, ilabel = load_patches_from_marked_images(marker_dir, IM_DIR, GT_DIR, kernel_size=kernel_size, use_lab=use_lab)

    save_patches(res_dir, patches, labels)







if __name__ == "__main__":

    split = 1


    for n_svox in [25]:
        for kernel_size in [3,5,9,11]:

            for dataset in [ 'refuge_std','refuge_new_match_rgb']:
                print(f"Getting patches from {dataset} {split} {n_svox} {kernel_size}")
                gen_patches(dataset, split, n_svox, kernel_size)