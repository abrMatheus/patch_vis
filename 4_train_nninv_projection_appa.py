import numpy as np
import matplotlib.pyplot as plt
import os
from src.nnp import NNP, NNPInv
import torch
from appa import appa as appa
from sklearn.manifold import TSNE
from src.functs import min_max_norm


seed = 42

import torch as T
T.manual_seed(42)
import random
random.seed(42)

np.random.seed(42)

def inverse_min_max(in_vec, ref_vec):

    ret_vec = in_vec*(ref_vec.max(0)-ref_vec.min(0) + 0.001)
    ret_vec = ret_vec + ref_vec.min(0)

    return ret_vec


def save_exp_nninv(exp_folder, nnp, appa,  pred_nnp, pred_appa, x_embedded, labels, patches, nnpinv, pred_nnpinv, nnpinv_appa, pred_appainv):
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    np.save(f"{exp_folder}/x_embedded.npy", x_embedded)
    np.save(f"{exp_folder}/pred_nnp.npy", pred_nnp)
    np.save(f"{exp_folder}/pred_appa.npy", pred_appa)
    np.save(f"{exp_folder}/label.npy", labels)
    np.save(f"{exp_folder}/patches.npy", patches)

    torch.save(nnp.state_dict(), f"{exp_folder}/nnp.pth")
    torch.save(appa._model.state_dict(), f"{exp_folder}/appa.pth")

    torch.save(nnpinv.state_dict(), f"{exp_folder}/nnpinv.pth")
    torch.save(nnpinv_appa.state_dict(), f"{exp_folder}/nnpinv_appa.pth")

    np.save(f"{exp_folder}/pred_nnpinv.npy", pred_nnpinv)
    np.save(f"{exp_folder}/pred_nnpinv_appa.npy", pred_appainv)


def load_exp(exp_folder):
    x_embedded = np.load(f"{exp_folder}/x_embedded.npy")
    pred_nnp = np.load(f"{exp_folder}/pred_nnp.npy")
    pred_appa = np.load(f"{exp_folder}/pred_appa.npy")
    labels = np.load(f"{exp_folder}/label.npy")
    patches = np.load(f"{exp_folder}/patches.npy")

    return x_embedded, pred_nnp, pred_appa, labels, patches



def normalize_patch(in_patches):
    ret_patches = (in_patches - in_patches.mean(0))
    ret_patches /= (in_patches.std(0) +0.001)

    return ret_patches


def inverse_norm(in_vec, ref_vec):
    ret_vec = (in_vec) * (ref_vec.std(0) +0.001)
    ret_vec = ret_vec + ref_vec.mean(0)

    return ret_vec

split = 1

metric="euclidean_distance"

datasets = ['plants','schisto','fish', 'refuge', 'gbm2d', 'mass_building']


alpha=1.0
spoint=1

for split in [1,2,3]:#,2,3]:
    for n_svox in [25]:#,50]:
        for k_size in [3,5,9]:#,5,9,11]:
            fig, axs = plt.subplots(len(datasets), 5, figsize=(30,13), constrained_layout=True)

            for i, dataset in enumerate(datasets):
                
                exp_folder_nnp = f"exps/{dataset}/s_{split}/n_{n_svox}/k_{k_size}"

                exp_folder = f"exps_nni/{dataset}/s_{split}/n_{n_svox}/k_{k_size}"

                x_embedded, pred_nnp, pred_appa, labels, patches = load_exp(exp_folder_nnp)
                nnp = NNP(patches.shape[1],2)
                state_dict = torch.load(f"{exp_folder_nnp}/nnp.pth")
                nnp.load_state_dict(state_dict)

                appa_model = appa.APPA(patches.shape[1], 2, kde_bandwidth=0.01)
                state_dict = torch.load(f"{exp_folder_nnp}/appa.pth")
                appa_model._model.load_state_dict(state_dict)


                n_patches = normalize_patch(patches)

                inv_model = NNPInv(2, n_patches.shape[1])

                # inv_model_appa = appa.APPA(2, n_patches.shape[1], kde_bandwidth=0.01)

                inv_model_appa = NNPInv(2, n_patches.shape[1])

                print("patches.shape", patches.shape, labels.shape)
                print("min max", patches.min(), patches.max())

                ## train models
                mm_n_patches = min_max_norm(n_patches)

                pred, x_emb, acc_loss = inv_model.fit(pred_nnp, mm_n_patches)
                _,_,_ = inv_model_appa.fit(pred_appa, mm_n_patches)

                # get predictions
                inv_model.eval()
                pred_nnp_nd  = inv_model.predict(pred_nnp)
                # pred_appa_nd = inv_model_appa.predict_no_grad(pred_appa).numpy()
                pred_appa_nd = inv_model_appa.predict(pred_appa)

                n_abs_nnp  = np.abs(mm_n_patches - pred_nnp_nd)

                n_abs_appa = np.abs(mm_n_patches - pred_appa_nd)



                pred_nnp_nd2 = inverse_min_max(pred_nnp_nd, n_patches)
                pred_appa_nd2 = inverse_min_max(pred_appa_nd, n_patches)

                # use predictions to get 2d
                pred_nnp_2   = nnp.predict_no_grad(pred_nnp_nd2).detach().cpu().numpy()
                pred_appa_2 = appa_model.predict_no_grad(pred_appa_nd2).detach().cpu().numpy()


                # print(n_patches.shape, pred_nnp.shape, pred_appa.shape, n_abs_nnp.shape, n_abs_appa.shape)

                mae_nnp  = np.mean(n_abs_nnp)
                mae_appa = np.mean(n_abs_appa)

                mae_nnp2  = np.mean(np.abs(n_patches - pred_nnp_nd2))
                mae_appa2 = np.mean(np.abs(n_patches - pred_nnp_nd2))

                pred_nnp_nd3 = inverse_norm(pred_nnp_nd2, patches)
                pred_appa_nd3 = inverse_norm(pred_appa_nd2, patches)

                mae_nnp3  = np.mean(np.abs(patches - pred_nnp_nd3))
                mae_appa3 = np.mean(np.abs(patches - pred_nnp_nd3))

                print(n_patches.shape, pred_nnp.shape, n_abs_nnp.shape, mae_nnp, mae_nnp2, mae_nnp3)
                print(n_patches.shape, pred_appa.shape, n_abs_appa.shape, mae_appa, mae_appa2, mae_appa3)

                axs[i][0].scatter(x_embedded[:,0],x_embedded[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)
                axs[i][1].scatter(pred_nnp[:,0],pred_nnp[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)
                axs[i][2].scatter(pred_nnp_2[:,0],pred_nnp_2[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)


                axs[i][3].scatter(pred_appa[:,0],pred_appa[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)
                axs[i][4].scatter(pred_appa_2[:,0],pred_appa_2[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=alpha, s=spoint)

                fig.suptitle(f"INV results for multiple datasets using svox={n_svox} k={k_size} and split {split} {metric}")
                fig.savefig(f"figs_nni/ntest_sv={n_svox}_k={k_size}_split={split}.png")


                fig2, axs2 = plt.subplots(5,10, figsize=(10,5))

                ni = np.random.randint(0, high=pred_nnp_nd.shape[0], size=25)


                pred_nnp_nd2   = inverse_norm(pred_nnp_nd2, patches)
                new_patches    = inverse_norm(n_patches, patches)

                pred_nnp_nd2 = min_max_norm(pred_nnp_nd2)
                new_patches  = min_max_norm(new_patches)

                # print(mm_n_patches.min(), mm_n_patches.max(), pred_nnp_nd.min(), pred_nnp_nd.max(), pred.min(), pred.max())
                print(new_patches.min(), new_patches.max())
                for i in range(25):
                    n = ni[i]

                    cx = i%5
                    cy = int(i/5)
                    
                    c = int(n_patches.shape[1]/(k_size*k_size))

                    # axs2[cx][cy+5].imshow(mm_n_patches[n].reshape(k_size,k_size,c))

                    # axs2[cx][cy].imshow(pred[n].reshape(k_size,k_size,c))


                    axs2[cx][cy+5].imshow(pred_nnp_nd2[n].reshape(k_size,k_size,c))

                    axs2[cx][cy].imshow(new_patches[n].reshape(k_size,k_size,c))

                fig2.savefig(f"figs_nni/test_i_{dataset}.png")


                save_exp_nninv(exp_folder, nnp, appa_model,  pred_nnp, pred_appa, x_embedded, labels, patches, inv_model, pred_nnp_nd, inv_model_appa, pred_appa_nd)
                # break
            # break
        # break
    # break
