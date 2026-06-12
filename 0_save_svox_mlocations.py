from src.functs import read_image,  save_marker, gen_svox_center
import os
import numpy as np
import pandas as pd
from CONFIG import *


def save_markers_dataset(dataset, split, n_svox):

    res_folder = f"s_markers/{dataset}/s_{split}/n_{n_svox}/"

    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    config_d = config[dataset]

    validation_file = f"{config_d['split_dir']}val{split}.csv"
    basename_list_val = list(pd.read_csv(validation_file, header=None)[0])
    basename_list_val = [i.replace('images/', '').replace('.png', '') for i in basename_list_val]


    GT_DIR = config_d["label_dir"]
    init_s = int(5*n_svox)
    sprefix = config_d["svox_dir_prefix"]
    SV_DIR = f"{sprefix}{init_s}_{n_svox}/"

    for i, basename in enumerate(basename_list_val):

        svox_img = read_image(f"{SV_DIR}/{basename}.png")

        gt_img = read_image(f"{GT_DIR}/{basename}.png")


        markers, mlabel = gen_svox_center(svox_img, gt_img)

        save_marker(markers, mlabel, f"{res_folder}/{basename}-seeds.txt", svox_img.shape)
        # break





if __name__ == "__main__":

    split = 1

    # for n_svox in [25,30,50]:
    #     for kernel_size in [3,5,9,11]:
    for n_svox in [25]:
        for kernel_size in [3]:
            for dataset in ['refuge_match']:
                save_markers_dataset(dataset, split, n_svox)