

import torch

import numpy as np
from src.nnp import NNP, NNPInv
import matplotlib.pyplot as plt

def load_exp_nninv(exp_folder_nnp):


    patches     =  np.load(f"{exp_folder_nnp}/patches.npy")
    labels      =  np.load(f"{exp_folder_nnp}/label.npy")

    nn_dict     = torch.load(f"{exp_folder_nnp}/nnp.pth")

    nn_inv_dict = torch.load(f"{exp_folder_nnp}/nnpinv.pth")


    return patches, labels, nn_dict, nn_inv_dict


def zcore_norm(in_vec, ref_vec):
    ret_vec = (in_vec - ref_vec.mean(0))
    ret_vec /= (ref_vec.std(0) +0.001)

    return ret_vec

def inverse_min_max(in_vec, ref_vec):

    ret_vec = in_vec*(ref_vec.max(0)-ref_vec.min(0) + 0.001)
    ret_vec = ret_vec + ref_vec.min(0)

    return ret_vec



dataset = "schisto"
k_size  = 5
n_svox  = 25
split   = 1

exp_folder_nnp = f"exps_nni/{dataset}/s_{split}/n_{n_svox}/k_{k_size}"


# read files
patches, labels, nn_dict, nn_inv_dict = load_exp_nninv(exp_folder_nnp)

nnp = NNP(patches.shape[1],2)

inv_model = NNPInv(2, patches.shape[1])


nnp.load_state_dict(nn_dict)
inv_model.load_state_dict(nn_inv_dict)

# plot proj

z_patches = zcore_norm(patches, patches)

x_emb_predic   = nnp.predict(z_patches)

#user clicks and show activation based on the click


click_2d = np.array([0.02,0.4]).reshape(1,2)

inv_model.eval()
click_nd = inv_model.predict(click_2d)


z_click_nd = inverse_min_max(click_nd, z_patches)


acts = np.matmul(z_patches, z_click_nd.T).reshape(-1)

acts[acts<0]=0

print(z_click_nd.shape, acts.shape, z_patches.shape)


fig, axs = plt.subplots(1,3)


axs[0].scatter(x_emb_predic[:,0],x_emb_predic[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=0.05)


axs[1].scatter(x_emb_predic[:,0],x_emb_predic[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=0.05)

axs[1].scatter(click_2d[0,0],click_2d[0,1], c='black', alpha=1.0)


axs[2].scatter(x_emb_predic[:,0],x_emb_predic[:,1], c=acts, cmap='inferno', alpha=0.05)

fig.savefig("test.png")
