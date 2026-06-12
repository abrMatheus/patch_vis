import torch

import numpy as np
from src.nnp import NNP, NNPInv
import matplotlib.pyplot as plt
import os
from skimage.draw import disk

from flim.experiments import utils

from pyflim import flim

import pandas as pd

def read_all_images(imlist, im_folder, use_lab=False):

    all_imgs = None
    all_imgs2 = None

    for i, iname in enumerate(imlist):
        basename = iname.split(".")[0]
        img = utils.load_image(f"{im_folder}/{basename}.png", lab=use_lab)

        img2 = utils.load_image(f"{im_folder}/{basename}.png", lab=False)

        img = np.expand_dims(img,0)
        img2 = np.expand_dims(img2,0)

        if all_imgs is None:
            all_imgs = img
            all_imgs2 = img2
        else:
            all_imgs = np.concatenate((all_imgs, img), axis=0)
            all_imgs2 = np.concatenate((all_imgs2, img2), axis=0)

        if i >10:
            break

    return all_imgs, all_imgs2


def get_best_act(all_images1, all_images2,imlist, centers, mdata, sdata, k_size, c_size, n_filter, use_mean=False):

    adjusted_c = centers/sdata
    bias = np.matmul(mdata.reshape(1,-1),adjusted_c.transpose())[0]
    # centers.shape, adjusted_c.shape, bias.shape

    u_centers = adjusted_c.reshape(n_filter,k_size,k_size,c_size)
    u_centers = torch.tensor(u_centers)
    u_centers = u_centers.permute(0,3,1,2)

    m = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5, stride=1, bias=False, padding=0)
    m.weight = torch.nn.Parameter(u_centers)
    m.bias = torch.nn.Parameter(-torch.tensor(bias))
    
    m = m.cuda()


    global_pool = torch.nn.AdaptiveMaxPool2d(1)
    if use_mean:
        global_pool = torch.nn.AdaptiveAvgPool2d(1)

    best_act = None
    best_act_v = -1000
    best_i = None
    best_name = None
    for i, img in enumerate(all_images1):
        with torch.no_grad():
            # img = utils.load_image(f"{imfolder}/{iname}.png", lab=True)
            img_t = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).cuda()
            res = m(img_t)
            res[res<=0]=0

            v = global_pool(res).detach().ravel()

            # print("v shape v value", v.shape, v, res.shape)
            if best_act_v < v.item():
                best_act_v = v.item()
                best_act = res.clone()
                best_i = all_images2[i]
                best_name = imlist[i]
            # break

    return best_act.detach().cpu().numpy()[0,0], best_i.copy(), best_name




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


validation_file = f"/dados/bases/schisto/csv_files/val1.csv"
basename_list_val = list(pd.read_csv(validation_file, header=None)[0])
basename_list_val = [i.replace('images/', '').replace('.png', '') for i in basename_list_val]



imfolder = "/dados/bases/schisto/orig_filtered/"
imlist = basename_list_val

all_imgs, all_imgs_o = read_all_images(basename_list_val,"/dados/bases/schisto/orig_filtered/", use_lab=True)


print(all_imgs.shape, all_imgs_o.shape)
# exit()

# read files
patches, labels, nn_dict, nn_inv_dict = load_exp_nninv(exp_folder_nnp)

mdata = patches.mean(0)
sdata = patches.std(0)

nnp = NNP(patches.shape[1],2)

inv_model = NNPInv(2, patches.shape[1])


nnp.load_state_dict(nn_dict)
inv_model.load_state_dict(nn_inv_dict)

# plot proj

z_patches = zcore_norm(patches, patches)

x_emb_predic   = nnp.predict(z_patches)

#user clicks and show activation based on the click


fig = plt.figure( figsize=(10,7))
fig.subplots_adjust(wspace=0.05, hspace=0.1)
gs = fig.add_gridspec(2,3)
ax_proj = fig.add_subplot(gs[:, 0])
ax_act   = fig.add_subplot(gs[:, 1])
ax_Iact  = fig.add_subplot(gs[0, 2])
ax_I  = fig.add_subplot(gs[1, 2])

ax_proj.set_title("proj.")
ax_proj.set_title("actv")

ax_Iact.set_title("act. img")
ax_I.set_title("highest act.")

plt.ion()

ax_proj.scatter(x_emb_predic[:,0],x_emb_predic[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=0.05, picker=True)

def onpick3(event):
    if event.mouseevent.button == 1: #left
        ind = event.ind[0]
        x = x_emb_predic[ind, 0]
        y = x_emb_predic[ind, 1]
        print('picked')
        print("index", ind)
        print("coords", x,y)

        ax_proj.clear()
        ax_act.clear()

        click_2d = np.array([x, y]).reshape(1,2)

        ax_proj.scatter(x_emb_predic[:,0],x_emb_predic[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=0.05, picker=True)
        ax_proj.scatter(click_2d[0,0],click_2d[0,1], c='black', alpha=1.0)

        inv_model.eval()
        click_nd = inv_model.predict(click_2d)
        z_click_nd = inverse_min_max(click_nd, z_patches)
        acts = np.matmul(z_patches, z_click_nd.T).reshape(-1)
        acts[acts<0]=0


        ax_act.scatter(x_emb_predic[:,0],x_emb_predic[:,1], c=acts, cmap='magma', alpha=0.1)

def onclick(event):
    # only inside axes
    # if event.inaxes == ax_proj:
    if event.inaxes != ax_proj:
        return
    if event.button !=1 :
        return
    print("x =", event.xdata)
    print("y =", event.ydata)

    click_2d = np.array([event.xdata,event.ydata]).reshape(1,2)
    
    ax_proj.clear()
    ax_act.clear()
    ax_Iact.clear()
    ax_I.clear()

    ax_proj.scatter(x_emb_predic[:,0],x_emb_predic[:,1], c=labels, cmap='tab10', vmin=1, vmax=10, alpha=0.05, picker=True)
    ax_proj.scatter(click_2d[0,0],click_2d[0,1], c='black', alpha=1.0)

    inv_model.eval()
    click_nd = inv_model.predict(click_2d)
    z_click_nd = inverse_min_max(click_nd, z_patches)
    acts = np.matmul(z_patches, z_click_nd.T).reshape(-1)
    acts[acts<0]=0


    ax_act.scatter(x_emb_predic[:,0],x_emb_predic[:,1], c=acts, cmap='magma', alpha=0.1)

    c_size=3
    n_filter=1
    centers = z_click_nd
    best_a, best_I, b_name = get_best_act(all_imgs, all_imgs_o, imlist, centers, mdata, sdata, k_size, c_size, n_filter)

    ax_Iact.set_title(b_name)
    ax_Iact.imshow(best_a)
    
    amax = np.argmax(best_a)
    coords = np.unravel_index(amax, best_a.shape) #TODO reshape img, maybe on conv or scale
    print(coords)
    rr, cc = disk((coords[0], coords[1]), 5)

    best_I[rr, cc,:] = [0,255,0]

    ax_I.imshow(best_I)

    ax_proj.set_title("proj.")
    ax_proj.set_title("actv")

    ax_Iact.set_title("act. img")
    ax_I.set_title("highest act.")




fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('pick_event', onpick3)

plt.show(block=True)