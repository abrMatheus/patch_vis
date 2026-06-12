import numpy as np
import matplotlib.pyplot as plt
import os
from src.nnp import NNP
import torch
from CONFIG import config
from src.flim_tools import load_patches_from_marked_images



def load_exp(exp_folder):
    if not os.path.exists(exp_folder):
        print("Not found folder", exp_folder)
        raise RuntimeError

    x_embedded = np.load(f"{exp_folder}/x_embedded.npy")
    pred = np.load(f"{exp_folder}/pred_nnp.npy")
    labels = np.load(f"{exp_folder}/label.npy")
    patches = np.load(f"{exp_folder}/patches.npy")

    state_dict = torch.load(f"{exp_folder}/nnp.pth")

    return state_dict, pred, patches, labels


def zcore_norm(in_vec, ref_vec):
    ret_vec = (in_vec - ref_vec.mean(0))
    ret_vec /= (ref_vec.std(0) +0.001)

    return ret_vec

split = 1

metric="euclidean_distance"

datasets = ['plants','schisto','fish', 'refuge', 'gbm2d', 'mass_building']

n_markers = 3
n_svox=25
k_size=3

fig, axs = plt.subplots(len(datasets), n_markers+1, figsize=(15,10))

for i, dataset in enumerate(datasets):

    exp_folder = f"exps/{dataset}/s_{split}/n_{n_svox}/k_{k_size}"

    s_dict, pred, patches, labels = load_exp(exp_folder)

    nnp = NNP(patches.shape[1])

    nnp.load_state_dict(s_dict)

    axs[i][0].scatter(pred[:,0],pred[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=0.2)

    for nm in range(n_markers):

        config_d  = config[dataset]
        GT_DIR = config_d["label_dir"]
        IM_DIR = config_d["orig_dir"]


        use_lab = False
        if config_d["color"]:
            use_lab=True

        marker_dir = f"markers/{dataset}/markers_s1_{nm+1}/"


        kernel_size = [k_size,k_size]
        m_patches, m_labels, m_ilabel = load_patches_from_marked_images(marker_dir, IM_DIR, GT_DIR, kernel_size=kernel_size, use_lab=use_lab)


        m_patches = zcore_norm(m_patches, patches)

        pred_m = nnp.predict(m_patches)

        axs[i][nm+1].scatter(pred[:,0],pred[:,1], c='gray', alpha=0.01)

        axs[i][nm+1].scatter(pred_m[:,0],pred_m[:,1], c=m_labels, cmap='tab10', vmin=1, vmax=10, alpha=0.2)

        axs[i][0].set_ylabel(dataset)

        axs[0][nm+1].set_title(f"markers {nm+1}")



        
axs[0][0].set_title("NNP")

plt.savefig("test_markers.png")