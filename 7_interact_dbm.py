import torch

import numpy as np
from src.nnp import NNP, NNPInv
from appa import appa as appa
import matplotlib.pyplot as plt
import os
from skimage.draw import disk

from src.functs import read_image, min_max_norm
from skimage.color import rgb2lab, gray2rgb
from src.flim_tools import my_patchify

import pandas as pd
import time

from matplotlib.widgets import Button
from dataclasses import dataclass
import tkinter as tk
from tkinter.simpledialog import askstring

from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances, cosine_distances

from CONFIG import config

from skimage.filters import threshold_otsu

from src.functs import save_marker



from skimage import io, measure
def filter_component_by_area(saliency, area_range=[2500,10000]):
    sal = saliency
    # bin_sal = np.copy(sal)
    bin_sal = sal
    # bin_sal[sal>threshold_otsu(saliency)] = 1
    # bin_sal[sal<=threshold_otsu(saliency)] = 0
    sal[bin_sal == 0] = 0
    
    sal_components_image = measure.label(bin_sal, background=0, connectivity=2)
    sal_nb_components = sal_components_image.max()
    
    bbs = []
    for c_ in range(1,sal_nb_components+1):
        area = len(sal_components_image[sal_components_image == c_])
        # print("area ", area)
        
        if (area < area_range[0] or area > area_range[1]):
            saliency[sal_components_image == c_] = 0



@dataclass
class SelectedPoint:
    x: float
    y: float
    label: str


def normalize_patch(in_patches, ref_vec):
    ret_patches = in_patches.copy()
    ret_patches = (ret_patches - ref_vec.mean(0))
    ret_patches /= (ref_vec.std(0) +0.001)

    return ret_patches

def inverse_min_max(in_vec, ref_vec):
    ret_vec = in_vec.copy()
    ret_vec = ret_vec*(ref_vec.max(0)-ref_vec.min(0) + 0.001)
    ret_vec = ret_vec + ref_vec.min(0)

    return ret_vec

def inverse_norm(in_vec, ref_vec):
    ret_vec = in_vec.copy()
    ret_vec = (ret_vec) * (ref_vec.std(0) +0.001)
    ret_vec = ret_vec + ref_vec.mean(0)

    return ret_vec


def my_dice(pred, gt):
    ab = np.count_nonzero(pred*gt)
    a  = np.count_nonzero(pred)
    b  = np.count_nonzero(gt)
    return 2*ab/(a+b)

def read_all_images(imlist, im_folder, gt_folder,use_lab=False):

    start = time.time()
    all_imgs = None
    all_limgs = None
    all_gtimg = None
    for i, iname in enumerate(imlist):
        basename = iname.split(".")[0]
        img = read_image(f"{im_folder}/{basename}.png")
        gt_i  = np.expand_dims(read_image(f"{gt_folder}/{basename}.png"), axis=0)
        if img.ndim == 3:
            lab_img = rgb2lab(img)
        else:
            #img = gray2rgb(img)
            lab_img = np.expand_dims(img.copy(), 2)


        img = np.expand_dims(img,0)
        lab_img = np.expand_dims(lab_img,0)

        if all_imgs is None:
            all_imgs = img
            all_limgs = lab_img
            all_gtimg = gt_i
        else:
            all_imgs = np.concatenate((all_imgs, img), axis=0)
            all_limgs = np.concatenate((all_limgs, lab_img), axis=0)

            all_gtimg = np.concatenate((all_gtimg, gt_i), axis=0)

        # if i >100:
            # break

    end = time.time()
    dur = end-start
    ratio=dur/i
    print(f"reading {dur:.3f} secs per {i} images {ratio:.3f} s/img")
    return all_imgs, all_limgs, all_gtimg


def get_best_act(all_imgs, all_limgs,imfolder,imlist, centers, mdata, sdata, k_size, n_filter, use_mean=False):

    c_size = centers.shape[1]
    c_size = int(c_size/(k_size*k_size))

    adjusted_c = centers/sdata
    bias = np.matmul(mdata.reshape(1,-1),adjusted_c.transpose())[0]
    # centers.shape, adjusted_c.shape, bias.shape

    u_centers = adjusted_c.reshape(n_filter,k_size,k_size,c_size)
    u_centers = torch.tensor(u_centers)
    u_centers = u_centers.permute(0,3,1,2)

    m = torch.nn.Conv2d(in_channels=c_size, out_channels=1, kernel_size=k_size, stride=1, bias=True, padding=0)
    m.weight = torch.nn.Parameter(u_centers)
    m.bias = torch.nn.Parameter(-torch.tensor(bias))
    
    m = m.cuda()


    best_act = None
    best_act_v = 1000
    best_i = None
    best_name = None
    best_vector= None
    start = time.time()
    # for i, iname in enumerate(imlist):
    for i, sample  in enumerate(zip(all_imgs, all_limgs)):
        with torch.no_grad():
            # img_raw = read_image(f"{imfolder}/{iname}.png")
            # img = rgb2lab(img_raw)
            img_raw, img = sample

            img_t = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).cuda().double()
            res = m(img_t)
            res[res<=0]=0

            p_img = my_patchify(img, patch_shape=[k_size,k_size]).reshape(-1,c_size*k_size*k_size)
            p_img = (p_img-mdata)/(sdata+0.001)

            dists = manhattan_distances(p_img, centers)

            max_i = np.argmin(dists)
            v = dists[max_i, 0]
            # print(p_img.dtype, img.dtype, centers.dtype, mdata.dtype)
            # print(v, max_i, centers[0,:5], p_img[max_i, :5], mdata[:5], sdata[:5], p_img.mean(0)[:5])
            if best_act_v > v:
                best_act_v = v.item()
                best_act = res.clone()
                best_i = img_raw.copy()
                best_name = imlist[i]
                best_coord = max_i
                best_vector = p_img[best_coord]
            # break
            # if i>10:
                # break
    print(best_act_v)
    end = time.time()
    dur = end-start
    ratio=dur/i
    print(f"{dur:.3f} secs per {i} images {ratio:.3f} s/img")
    return best_act.detach().cpu().numpy()[0,0], best_i.copy(), best_name, best_coord, best_vector



def load_exp_nninv(exp_folder_nnp):


    patches     =  np.load(f"{exp_folder_nnp}/patches.npy")
    labels      =  np.load(f"{exp_folder_nnp}/label.npy")   

    nn_dict     = torch.load(f"{exp_folder_nnp}/nnp.pth")

    nn_inv_dict = torch.load(f"{exp_folder_nnp}/nnpinv.pth")

    appa_dict     = torch.load(f"{exp_folder_nnp}/appa.pth")

    appa_inv_dict = torch.load(f"{exp_folder_nnp}/nnpinv_appa.pth")


    return patches, labels, nn_dict, nn_inv_dict, appa_dict, appa_inv_dict

def maybe_resize(x, shape):
    if x.shape[-2:] != shape[-2:]:
        x = torch.nn.functional.interpolate(x, size=shape[-2:],
                        mode="bilinear", align_corners=True)
    return x

class PointExplorer:
    def __init__(self, dataset, k_size=3, n_svox=25, split=1, use_appa=True, mark_norm=False, use_patch=False):
        self.dataset = dataset
        self.k_size  = k_size
        self.n_svox  = n_svox
        self.split   = split

        self.use_appa=use_appa
        self.mark_norm=mark_norm
        self.use_patch=use_patch # TODO:


        ################ trained files
        self.allimgs = None
        self.all_limgs = None
        self.nnp  = None
        self.inv_model = None
        self.x_emb_predic = None
        self.imfolder = None
        self.imlist = None
        self.z_patches = None
        self.labels = None



        # interactive data
        self.selected_points = []
        self.current_marker = None
        self.selected_centers = None
        self.markers = []


        self.torch_img_model = None
        self.torch_dbm_model = None
        self.torch_img_dec = None
        self.torch_dbm_dec = None
        self.dbm_nd = None
        self.dbm_2d = None


        self.fig = plt.figure(figsize=(20,7))

        self.fig.subplots_adjust(wspace=0.05, hspace=0.1)
        gs = self.fig.add_gridspec(2,3)
        self.ax_proj = self.fig.add_subplot(gs[:, 0])
        self.ax_act     = self.fig.add_subplot(gs[0, 1])
        self.ax_dbm     = self.fig.add_subplot(gs[1, 1])
        self.ax_Iact  = self.fig.add_subplot(gs[0, 2])
        self.ax_I  = self.fig.add_subplot(gs[1, 2])

        self.ax_proj.set_title("proj. (click here)")
        self.ax_act.set_title("actv")
        self.ax_dbm.set_title("actv")
        self.ax_Iact.set_title("act. img")
        self.ax_I.set_title("highest act.")

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        # self.fig.canvas.mpl_connect('pick_event', self.onpick3) #TODO: select exact point

        # Done button
        ax_button = plt.axes([0.82, 0.02, 0.12, 0.05])
        self.done_button = Button(ax_button, "Done")
        self.done_button.on_clicked(self.finish)

        self.prepare_data()
    
    def prepare_data(self):
        exp_folder_nnp = f"exps_nni/{self.dataset}/s_{self.split}/n_{self.n_svox}/k_{self.k_size}"

        configd = config[self.dataset]

        split_dir = configd["split_dir"]
        validation_file = f"{split_dir}/val1.csv"
        basename_list_val = list(pd.read_csv(validation_file, header=None)[0])
        basename_list_val = [i.replace('images/', '').replace('.png', '') for i in basename_list_val]


        orig_dir = configd["orig_dir"]
        self.imfolder = f"{orig_dir}"
        self.imlist = basename_list_val
        self.gt_folder = configd["label_dir"]


        self.allimgs, self.all_limgs, self.gt_imgs = read_all_images(self.imlist, self.imfolder, self.gt_folder)

        patches, self.labels, nn_dict, nn_inv_dict, appa_dict, appa_inv_dict = load_exp_nninv(exp_folder_nnp)

        self.mdata = patches.mean(0)
        self.sdata = patches.std(0)

        if self.use_appa:
            self.p_model  = appa.APPA(patches.shape[1],2)
            self.inv_model = NNPInv(2, patches.shape[1])

            self.p_model._model.load_state_dict(appa_dict)
            self.inv_model.load_state_dict(appa_inv_dict)
        
        else:
            # self.p_model = NNP(patches.shape[1],2)
            self.inv_model = NNPInv(2, patches.shape[1])

            self.p_model.load_state_dict(nn_dict)
            self.inv_model.load_state_dict(nn_inv_dict)

        # plot proj

        self.z_patches = normalize_patch(patches, patches)

        self.x_emb_predic   = self.p_model.predict_no_grad(self.z_patches).detach().numpy()

        self.ax_proj.scatter(self.x_emb_predic[:,0],self.x_emb_predic[:,1], c=self.labels, cmap='tab10', vmin=1, vmax=10, alpha=1.0, s=3, picker=True)


        # dbm
        Npoints=51
        s_x = np.linspace(0, 1, Npoints)
        s_y = np.linspace(0, 1, Npoints)
        # full coordinate arrays
        xx, yy = np.meshgrid(s_x, s_y)
        self.dbm_nd = torch.zeros((1,Npoints, Npoints, self.z_patches.shape[1]))

        self.mesh_2d = np.concatenate((xx.reshape(Npoints,Npoints,1), yy.reshape(Npoints,Npoints,1)), axis=2)
        for i in range(0,Npoints):
            for j in range(0,Npoints):
                x,y = self.mesh_2d[i,j]

                click_2d = np.array([x,y]).reshape(1,2)
                click_nd = self.inv_model.predict(click_2d)
                z_click_nd = inverse_min_max(click_nd ,self.z_patches)
                self.dbm_nd[0,i,j,:] = torch.tensor(z_click_nd[0,:]) #todo: otimizar isso, conversao torch-np-torch

        self.dbm_nd = self.dbm_nd.permute(0,3,1,2).float().cuda()


    def on_click(self, event):
        if event.inaxes != self.ax_proj:
            return
        if event.button !=1 :
            return
        
        x = event.xdata
        y = event.ydata
        print(f"x,y = [{x:.2f}, {y:.2f}]")

        click_2d = np.array([event.xdata,event.ydata]).reshape(1,2)

        self.ax_act.clear()
        self.ax_Iact.clear()
        self.ax_I.clear()
        

        if self.current_marker is not None:
            self.current_marker.remove()

        self.current_marker = self.ax_proj.scatter(
            x, y, color="black", s=40, marker="*", #zonder=5
        )
        self.fig.canvas.draw_idle()

        self.inv_model.eval()
        click_nd = self.inv_model.predict(click_2d)
        z_click_nd = inverse_min_max(click_nd ,self.z_patches)
        acts = np.matmul(self.z_patches, z_click_nd.T).reshape(-1)
        print("Acts min max ", acts.min(), acts.max())
        acts[acts<0]=0

        
        self.ax_act.scatter(self.x_emb_predic[:,0],self.x_emb_predic[:,1], c=acts, cmap='magma', alpha=1.0, s=3)

        n_filter=1
        centers = z_click_nd
        best_a, best_I, b_name, best_coord, best_vector = get_best_act(self.allimgs, self.all_limgs,self.imfolder, self.imlist, centers, self.mdata, self.sdata, self.k_size, n_filter)
    
        # caso queira mostrar act do patch
        # acts = np.matmul(self.z_patches, best_vector.T).reshape(-1)
        # acts[acts<0]=0
        # self.ax_dbm.clear()
        # self.ax_dbm.scatter(self.x_emb_predic[:,0],self.x_emb_predic[:,1], c=acts, cmap='magma', alpha=1.0, s=3)
        
        self.ax_Iact.set_title(b_name)
        self.ax_Iact.imshow(best_a, cmap="magma")
        
        coords = [int(best_coord/best_I.shape[1]), best_coord%best_I.shape[1]]
        if coords[0] >= 4 and coords[0]<=best_I.shape[0]-4 and coords[1] >= 4 and coords[1]<=best_I.shape[1]-4:
            rr2, cc2 = disk((coords[0], coords[1]), 3)
            rr3, cc3 = disk((coords[0], coords[1]), 4)
        else:
            rr2, cc2 = disk((coords[0], coords[1]), 1)
            rr3, cc3 = disk((coords[0], coords[1]), 1)

        if best_I.ndim==2:
            best_I = gray2rgb(best_I)
        best_I[rr3, cc3,:] = [0,0,0]
        best_I[rr2, cc2,:] = [255,255,0]


        self.ax_I.imshow(best_I)
        self.ax_act.set_title("actv")
        self.ax_Iact.set_title("act. img")
        self.ax_I.set_title("highest act.")

        # self.fig.canvas.draw_idle()
        # plt.pause(0.05)

        self.fig.canvas.draw()      # force immediate redraw
        self.fig.canvas.flush_events()

        # Ask for annotation
        root = tk.Tk()
        root.withdraw()

        label = askstring(
            "Point Annotation",
            f"Annotation for ({x:.2f}, {y:.2f}):"
        )

        root.destroy()

        if label is None:
            self.ax_act.clear()
            self.ax_dbm.clear()
            self.ax_Iact.clear()
            self.ax_I.clear()

            self.fig.canvas.draw_idle()
            return

        # Store point
        self.selected_points.append(
            SelectedPoint(x=x, y=y, label=label)
        )

        if self.selected_centers is None:
            self.selected_centers = centers
        else:
            self.selected_centers = np.concatenate((self.selected_centers, centers), axis=0)
        self.markers.append([b_name, coords, int(label)])

        # Permanently annotate plot

        self.ax_proj.scatter(x, y, color=["red","green"][int(label)], s=50, marker="*")
        self.ax_proj.text(
            x+0.005, y+0.005, label,
            fontsize=15,
            color=["red","green"][int(label)]
        )

        # update model
        self.update_models() #update torch and dbm models based centers and labels

        print(self.dbm_nd.dtype, self.dbm_nd.shape, self.torch_dbm_model)
        print(self.torch_img_dec)

        self.torch_dbm_model.cuda()
        self.torch_dbm_model.eval()
        self.torch_img_dec.cuda()
        self.torch_img_dec.eval()
        tmp = self.torch_dbm_model(self.dbm_nd)
        print("out models", tmp.shape)
        tmp[tmp<0]=0 #RELU
        tmp = self.torch_img_dec(tmp)
        print("out models", tmp.shape)

        dec = tmp.detach().cpu().numpy()[0,0]

        print("DECS min max ", dec.min(), acts.max())

        # dec=dec/(np.abs(dec.max())+0.01) # decoder retorna valores entre 0,255 e nao binarios
        # dec = (dec - dec.min())/(dec.max() - dec.min() + 0.001)
        thold = threshold_otsu(dec)
        print("thold", thold)
        mask = dec>thold
        dec[mask==False] = 0
        dec[mask==True] = 1

        self.ax_dbm.clear()

        self.ax_dbm.imshow(dec, cmap="tab10", extent=[0, 1, 0, 1], origin='lower', alpha=0.2, vmax=9, vmin=0)

        # todo: show a dice score
        # self.ax_dbm.imshow(dec, cmap="tab10", extent=[0, 1, 0, 1], origin='lower', vmin=0, vmax=9, alpha=0.2)
        


        print(np.unique(dec), np.unique(self.labels))

        self.ax_dbm.scatter(self.x_emb_predic[:,0],self.x_emb_predic[:,1], c=self.labels, cmap='tab10', alpha=1.0, s=1, vmin=1, vmax=10)



        self.fig.canvas.draw_idle()


    def update_models(self):
        f_size = self.selected_centers.shape[1]
        c_size = int(f_size/(self.k_size*self.k_size))
        o_size = self.selected_centers.shape[0]


        if self.mark_norm:
            print("MARKNORMMMMMMMMMMMMMMMMMM")
            use_mean = self.selected_centers.mean(0)
            use_std  = self.selected_centers.std(0)

            use_kernels = (self.selected_centers - use_mean)/(use_std+0.001)

        else:
            use_mean = self.mdata
            use_std  = self.sdata

            use_kernels = self.selected_centers.copy()



        if self.torch_img_model is not None:
            del self.torch_img_model
            del self.torch_dbm_model
            del self.torch_img_dec
            del self.torch_dbm_dec
            
        self.torch_img_model = torch.nn.Conv2d(in_channels=c_size, out_channels=o_size, kernel_size=self.k_size, stride=1, bias=True, padding=0)
        self.torch_img_dec = torch.nn.Conv2d(in_channels=o_size, out_channels=1, kernel_size=1, stride=1, bias=False, padding=0)
        self.torch_dbm_model = torch.nn.Conv2d(in_channels=self.selected_centers.shape[1], out_channels=o_size, kernel_size=1, stride=1, bias=True, padding=0)
        self.torch_dbm_dec = torch.nn.Conv2d(in_channels=o_size, out_channels=1, kernel_size=1, 
        stride=1, bias=False, padding=0)


        # TODO: ver questao de 

        kernels_dbm = use_kernels.reshape(o_size,1,1,f_size).copy()
        kernels_dbm = torch.tensor(kernels_dbm).permute(0,3,1,2)

        use_kernels = use_kernels/(use_std+0.001)
        use_bias    = np.matmul(use_mean.reshape(1,-1), use_kernels.transpose())[0]

        kernels_img = use_kernels.reshape(o_size,self.k_size,self.k_size,c_size)
        kernels_img = torch.tensor(kernels_img).permute(0,3,1,2)


        self.torch_img_model.weight = torch.nn.Parameter(kernels_img)
        self.torch_img_model.bias = torch.nn.Parameter(-torch.tensor(use_bias))

        self.torch_dbm_model.weight = torch.nn.Parameter(kernels_dbm.float())
        self.torch_dbm_model.bias = torch.nn.Parameter(-torch.tensor(use_bias).float())

        labels = []
        for i in range(o_size):
            l = self.markers[i][2]
            if l>0:
                l= 1
            else:
                l=-1
            labels.append(l)


        dec_weights = torch.nn.Parameter(torch.tensor(labels).reshape(1,o_size,1,1).float())
        self.torch_img_dec.weight = torch.nn.Parameter(dec_weights)
        print("dec weights", dec_weights)
        # self.torch_dbm_dec.weight = torch.nn.Parameter(dec_weights)
        


    def finish(self, event):
        print("FINISH")
        print("\nCollected points:\n")

        kernel_labels = []
        for i, p in enumerate(self.selected_points, start=1):
            print(
                f"{i}: x={p.x:.5f}, "
                f"y={p.y:.5f}, "
                f"label='{p.label}'"
            )
            kernel_labels.append(int(p.label))
        
        self.run_pipeline(kernel_labels)


    def run_pipeline(self, kernel_labels):

        self.torch_img_model.cuda()

        o_size = self.selected_centers.shape[0]

        mean_dice = 0.0
        mean_fdice = 0.0
        Nims = len(self.gt_imgs)
        for i, sample  in enumerate(zip(self.allimgs, self.all_limgs, self.gt_imgs)):
            with torch.no_grad():
                img_raw, img, gt_img = sample
                img_t = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).cuda().double()
                res = self.torch_img_model(img_t)
                res[res<=0]=0 # relu

                res = maybe_resize(res, img_t.shape)

                # print(torch.amax(res, dim=(2,3)), res.shape)

                sal = self.torch_img_dec(res.float())[0,0]
                sal = sal.detach().cpu().numpy()
                thold = threshold_otsu(sal)
                sal[sal<=thold] = 0
                sal[sal>thold] = 1

                #decode
                # sal = self.decode_by_ldecoder(res, img.shape[0:2],torch.tensor(kernel_labels).cuda())

                dice = my_dice(sal,gt_img)

                mean_dice += dice/Nims

                if self.dataset == "schisto":
                    filter_component_by_area(sal, area_range=[1000,9000])
                    dice = my_dice(sal,gt_img)

                    mean_fdice += dice/Nims

        #         print(sal.shape, gt_img.shape, dice)

        #         break

        fig2, axs2 = plt.subplots(1, o_size + 3)

        for i in range(o_size):
            axs2[i+1].imshow(res[0,i].detach().cpu().numpy(), cmap="inferno")
            axs2[i+1].set_title(f"act{i}")
        
        axs2[-2].imshow(sal, cmap="gray")
        axs2[-1].imshow(gt_img, cmap="gray")
        axs2[0].imshow(img_raw)

        axs2[0].set_title("Img")
        axs2[-2].set_title("Pred.")
        axs2[-1].set_title("GT")

        plt.show()

        print(f"MEAN DICE IS {mean_dice:.3f}  {mean_fdice:.3f}")


        # last_folder_i = 1
        # exp_name = f"exp_interact/{self.dataset}/"
        # if os.path.exists(exp_name):
        #     ifolder = os.listdir(exp_name)
        #     if len(ifolder)>0:
        #         last_folder_i = int(ifolder[-1]) +1

        # os.makedirs(f"{exp_name}/{last_folder_i}/markers")

        # f = open(f"{exp_name}/{last_folder_i}/dice.txt", "w")
        # line = f"MEAN DICE IS {mean_dice:.3f}  {mean_fdice:.3f}\n"
        # f.writelines([line])

        # for i in range(len(self.markers)):
        #     coords = [self.markers[i][1]]
        #     mname   = self.markers[i][0]
        #     mlabel  = [kernel_labels[i]]
        #     print("mname", "coords", "mlabel", mname, coords, mlabel)
        #     path = f"{exp_name}/{last_folder_i}/markers/{mname}-seeds.txt"
        #     save_marker(coords, mlabel, path, self.all_limgs[0].shape)
        


if __name__ == "__main__":

    explorer = PointExplorer(dataset="fish", k_size=3, n_svox=25, split=1)
    plt.show(block=True)