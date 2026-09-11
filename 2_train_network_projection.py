import numpy as np
import matplotlib.pyplot as plt
import os
from src.nnp import NNP
import torch
from appa import appa as appa
from sklearn.manifold import TSNE, MDS
from src.functs import min_max_norm

import umap

seed = 42

import torch as T
T.manual_seed(42)
import random
random.seed(42)

np.random.seed(42)


def save_exp(exp_folder, nnp, appa,  pred_nnp, pred_appa, x_embedded, labels, patches):
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    np.save(f"{exp_folder}/x_embedded.npy", x_embedded)
    np.save(f"{exp_folder}/pred_nnp.npy", pred_nnp)
    np.save(f"{exp_folder}/pred_appa.npy", pred_appa)
    np.save(f"{exp_folder}/label.npy", labels)
    np.save(f"{exp_folder}/patches.npy", patches)

    torch.save(nnp.state_dict(), f"{exp_folder}/nnp.pth")
    torch.save(appa.state_dict(), f"{exp_folder}/appa.pth")
    

def normalize_patch(in_patches):
    ret_patches = (in_patches - in_patches.mean(0))
    ret_patches /= in_patches.std(0)

    return ret_patches

split = 1

kde_bandwidth=0.01

metric="euclidean_distance"

# datasets = ['plants','schisto','fish', 'gbm2d', 'mass_building']

datasets = [ 'plants','schisto','refuge', 'fish', 'gbm2d', 'mass_building']

# datasets = ["brats2d"]

# datasets = ['refuge', 'refuge_rgb', 'refuge_new_match', 'refuge_new_match_rgb', 'refuge_std']

for split in [1,2,3]:#,2,3]:
    for n_svox in [25]:#,50]:
        alpha=1.0
        spoint = 1
        for k_size in [3,5,9]:#,5,9,11]:
            fig, axs = plt.subplots(len(datasets)+1, 3, figsize=(15,10), constrained_layout=True)

            for i, dataset in enumerate(datasets):
                
                patch_dir = f"s_patches/{dataset}/s_{split}/n_{n_svox}/k_{k_size}"


                patches = np.load(f"{patch_dir}/not_norm_patches_l1.npy")
                labels  = np.load(f"{patch_dir}/labels.npy")

                n_patches = normalize_patch(patches)
                # n_patches = patches.copy()

                print("patches.shape", n_patches.shape)

                x_embedded = TSNE(n_components=2, learning_rate='auto',init='random',
                                perplexity=20, random_state=42).fit_transform(n_patches)

                # x_embedded = umap.UMAP(random_state=42).fit_transform(n_patches)
                # x_embedded = MDS(n_components=2, random_state=42, max_iter=1, init="classical_mds").fit_transform(n_patches)
                
                # continue
                x_embedded = min_max_norm(x_embedded)

                model = NNP(n_patches.shape[1], 2, lr=0.001)

                model_appa = appa.APPA(n_patches.shape[1], 2, kde_bandwidth=kde_bandwidth)

                print("patches.shape", patches.shape, labels.shape)
                pred, x_emb, acc_loss = model.fit(n_patches, x_embedded, batch_size=128)
                model_appa.fit(n_patches, x_embedded)

                pred_nnp  = model.predict_no_grad(n_patches).detach().cpu().numpy()

                pred_appa = model_appa.predict_no_grad(n_patches).detach().cpu().numpy()


                axs[i][0].scatter(x_embedded[:,0],x_embedded[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)
                axs[i][1].scatter(pred_nnp[:,0],pred_nnp[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)
                axs[i][2].scatter(pred_appa[:,0],pred_appa[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)

                axs[i][0].set_title("t-SNE projection")
                axs[i][1].set_title("NNP")
                axs[i][2].set_title("APPA")

                axs[i][0].set_ylabel(f"{dataset}")

                exp_folder = f"exps/{dataset}/s_{split}/n_{n_svox}/k_{k_size}"

                # model_appa.trainer.save_checkpoint("path/to/save/model.ckpt")

                save_exp(exp_folder, model, model_appa._model, pred_nnp, pred_appa, x_embedded, labels, patches)




            if not os.path.exists("results/"):
                os.makedirs("results/")
            
            fig.suptitle(f"appa results for multiple datasets using svox={n_svox} k={k_size} and split {split} {metric} - kde_bandwidth {kde_bandwidth}")
            fig.savefig(f"results/nnp_appa_sv={n_svox}_k={k_size}_split={split}.png")

            # break
        # break
    # break