import numpy as np
import matplotlib.pyplot as plt
import os
from src.nnp import NNP, NNPInv
import torch
from appa import appa as appa
from sklearn.manifold import TSNE,MDS
from src.functs import min_max_norm
import umap

seed = 42

torch.manual_seed(42)
import random
random.seed(42)

np.random.seed(42)


class MyEarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.min_val_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        # if (validation_loss - train_loss) > self.min_delta:
        #     self.counter +=1
        #     if self.counter >= self.tolerance:  
        #         self.early_stop = True

        if validation_loss > self.min_val_loss + self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True

        if validation_loss < self.min_val_loss:
            self.min_val_loss = validation_loss
            self.counter = 0

def inverse_min_max(in_vec, ref_vec):

    ret_vec = in_vec*(ref_vec.max(0)-ref_vec.min(0) + 0.001)
    ret_vec = ret_vec + ref_vec.min(0)

    return ret_vec

def normalize_patch(in_patches):
    ret_patches = (in_patches - in_patches.mean(0))
    ret_patches /= in_patches.std(0)

    return ret_patches


def save_models_targets_and_gt(exp_folder, patches, labels, x_embedded, nnp_model, appa_model, inv_model, inv_appa_model):
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    
    np.save(f"{exp_folder}/x_embedded.npy", x_embedded)
    np.save(f"{exp_folder}/label.npy", labels)
    np.save(f"{exp_folder}/patches.npy", patches)

    torch.save(nnp_model.state_dict(), f"{exp_folder}/nnp.pth")
    torch.save(appa_model._model.state_dict(), f"{exp_folder}/appa.pth")

    torch.save(inv_model.state_dict(), f"{exp_folder}/nnpinv.pth")
    torch.save(inv_appa_model.state_dict(), f"{exp_folder}/nnpinv_appa.pth")



def save_pred_i(exp_folder, i_sufix, pred_nnp, pred_appa):
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    np.save(f"{exp_folder}/pred_nnp_{i_sufix}.npy", pred_nnp)
    np.save(f"{exp_folder}/pred_appa_{i_sufix}.npy", pred_appa)




from sklearn.model_selection import train_test_split


split=1
n_svox=25
k_size=3
kde_bandwidth=0.01

alpha=1.0
spoint=1

datasets = ['plants','schisto','fish', 'refuge', 'gbm2d', 'mass_building']
# datasets = ['plants','fish']

# datasets = ['brats2d']

for proj_method in ["t-SNE"]:#, "UMAP", "MDS"]:
    fig, axs = plt.subplots(len(datasets), 5, figsize=(5*3,3*len(datasets)), constrained_layout=True)
    fig2, axs2 = plt.subplots(1, 3, figsize=(9, 4), constrained_layout=True)
    for i, dataset in enumerate(datasets):
        patch_dir = f"s_patches/{dataset}/s_{split}/n_{n_svox}/k_{k_size}"

        patches = np.load(f"{patch_dir}/not_norm_patches_l1.npy")
        labels  = np.load(f"{patch_dir}/labels.npy")


        # patches, _, labels, _ = train_test_split(patches, labels, train_size=3000, stratify=labels)

        n_patches = normalize_patch(patches)

        if proj_method == "t-SNE":
            x_embedded = TSNE(n_components=2, perplexity=20, learning_rate='auto',init='random',random_state=42).fit_transform(n_patches)
        elif proj_method == "UMAP":
            x_embedded = umap.UMAP(random_state=42).fit_transform(n_patches)
        elif proj_method == "MDS":
            x_embedded = MDS(n_components=2, random_state=42, max_iter=1, init="classical_mds").fit_transform(n_patches)
        else:
            raise NotImplemented

        x_embedded = min_max_norm(x_embedded)

        ## TRAIN NNP - APPA ################################

        nnp_model = NNP(n_patches.shape[1], 2, lr=0.001)

        appa_model = appa.APPA(n_patches.shape[1], 2, kde_bandwidth=kde_bandwidth, training_epochs=300)

        pred, x_emb, acc_loss = nnp_model.fit(n_patches, x_embedded, batch_size=1024, epochs=300)
        appa_model.fit(n_patches, x_embedded)


        pred_nnp  = nnp_model.predict_no_grad(n_patches).detach().cpu().numpy()

        pred_appa = appa_model.predict_no_grad(n_patches).detach().cpu().numpy()


        ## Train inverse NNP - APPA

        inv_model = NNPInv(2, n_patches.shape[1])
        inv_appa_model = NNPInv(2, n_patches.shape[1])

        mm_n_patches = min_max_norm(n_patches)

        pred, x_emb, acc_loss = inv_model.fit(pred_nnp, mm_n_patches, batch_size=1024,epochs=300)
        _,_,_ = inv_appa_model.fit(pred_appa, mm_n_patches, batch_size=1024, epochs=300)


        pred_nnp_nd  = inv_model.predict(pred_nnp)
        pred_appa_nd = inv_appa_model.predict(pred_appa)

        pred_nnp_nd2 = inverse_min_max(pred_nnp_nd, n_patches)
        pred_appa_nd2 = inverse_min_max(pred_appa_nd, n_patches)
    
        pred_nnp_2   = nnp_model.predict_no_grad(pred_nnp_nd2).detach().cpu().numpy()
        pred_appa_2 = appa_model.predict_no_grad(pred_appa_nd2).detach().cpu().numpy()


        axs[i][0].scatter(x_embedded[:,0],x_embedded[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)
        axs[i][1].scatter(pred_nnp[:,0],pred_nnp[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)
        axs[i][2].scatter(pred_nnp_2[:,0],pred_nnp_2[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)


        axs[i][3].scatter(pred_appa[:,0],pred_appa[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)
        axs[i][4].scatter(pred_appa_2[:,0],pred_appa_2[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)


        axs[i][0].set_title("{proj_method} projection")
        axs[i][1].set_title("NNP")
        axs[i][2].set_title("NNP->NNInv->NNP")


        axs[i][3].set_title("APPA")
        axs[i][4].set_title("APPA->NNInv->APPA")

        axs[i][0].set_ylabel(f"{dataset}")

        exp_folder = f"ext_appa_ok/{proj_method.lower()}/{dataset}/"

        save_models_targets_and_gt(exp_folder, patches, labels, x_embedded, nnp_model, appa_model, inv_model, inv_appa_model)


        save_pred_i(exp_folder, "1_0", pred_nnp, pred_appa)
        save_pred_i(exp_folder, "2_0", pred_nnp_2, pred_appa_2)


        x_nnp_nd  = n_patches.copy()
        x_appa_nd = n_patches.copy()

        for pred_i in range(1,11):
            print("STARTING ", pred_i)

            
            pred_nnp  = nnp_model.predict_no_grad(x_nnp_nd).detach().cpu().numpy()
            pred_appa = appa_model.predict_no_grad(x_appa_nd).detach().cpu().numpy()

            pred_nnp_nd  = inv_model.predict(pred_nnp)
            pred_appa_nd = inv_appa_model.predict(pred_appa)

            del x_nnp_nd, x_appa_nd

            x_nnp_nd = inverse_min_max(pred_nnp_nd, n_patches)
            x_appa_nd = inverse_min_max(pred_appa_nd, n_patches)


            save_pred_i(exp_folder, f"{pred_i}", pred_nnp, pred_appa)

            axs2[0].scatter(x_embedded[:,0],x_embedded[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)
            axs2[1].scatter(pred_nnp[:,0],pred_nnp[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)
            axs2[2].scatter(pred_appa[:,0],pred_appa[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)
            fig2.savefig(f"ext_appa_ok/{proj_method.lower()}/{dataset}/{pred_i}.png")

            axs2[0].clear()
            axs2[1].clear()
            axs2[2].clear()

    fig.suptitle(f"appa results for multiple datasets using svox={n_svox} k={k_size} and split {split} - kde_bandwidth {kde_bandwidth}")
    fig.savefig(f"ext_appa_ok/{proj_method.lower()}/seg_br_nnp_appa_sv={n_svox}_k={k_size}_split={split}.png")



    # break 

