import torch

import os
from pyflim import arch, flim
from flim.experiments import utils
import numpy as np

from src.functs import read_image, creat_marker_label_image, read_markers


def resize(y_act, original_size=[400,400]):
    y_act = torch.tensor(y_act).permute(2,0,1).unsqueeze(0)
    out_size = y_act[0].shape[-2:]
    if(out_size[0] != original_size[0] or out_size[1] != original_size[1]):
        y_act = torch.nn.functional.interpolate(y_act, [original_size[0], original_size[1]], mode='bilinear', align_corners=True)#.squeeze(0)
    return y_act.permute(0,2,3,1).detach().numpy()[0]


def load_patches_from_marked_images(mark_dir, im_dir, gt_dir, kernel_size=[3,3], mimage_dir=None, use_lab=True):

    imlist = os.listdir(mark_dir)

    ret_patches  = None
    ret_labels   = None
    ret_imlabel  = None

    ret_mpatches = None
    ret_names = [ ]

    Nim = len(imlist)
    for i, mi in enumerate(imlist):
        print(f"Images {i}/{Nim}", flush=True, end='\r')
        basename = mi.split("-seeds")[0]

        # img = read_image(f"{im_dir}/{basename}.png")
        img = utils.load_image(f"{im_dir}/{basename}.png", lab=use_lab) # todo: nem sei se é equivalente
        # img = read_image(f"{im_dir}/{basename}.png")
        # img = _image_to_lab(img)
        gti = read_image(f"{gt_dir}/{basename}.png")


        markers, mlabel = read_markers(f"{mark_dir}/{mi}")

        marker_image = creat_marker_label_image(gti, markers, mlabel)





        patch_i = flim.FLIMModel.patchify(img, patch_shape=kernel_size)
        patch_m = flim.FLIMModel.patchify(marker_image, patch_shape=[1,1]).reshape(-1)


        mpatch_m = patch_m[patch_m>0]
        mpatch_i = patch_i[patch_m>0]

        N = mpatch_m.shape[0]

        mpatch_i = mpatch_i.reshape(N,-1)
        mpatch_m = mpatch_m.reshape(N,-1)


        if mimage_dir is not None:
            mimg = utils.load_mimage(f"{mimage_dir}/{basename}.mimg")
            mimg = resize(mimg, original_size=gti.shape[0:2])
            # patch_mi = flim.FLIMModel.patchify(mimg, patch_shape=[1,1])

            patch_mi = flim.FLIMModel.patchify(mimg, patch_shape=kernel_size)

            mpatch_mi = patch_mi[patch_m>0]
            mpatch_mi = mpatch_mi.reshape(N,-1)

            if i==0:
                ret_mpatches = mpatch_mi
            else:
                ret_mpatches = np.concatenate((ret_mpatches, mpatch_mi), axis=0)



        # ret_patches.append(mpatch_i)
        # ret_labels.append(mpatch_m)

        ret_names.append(basename)

        if i==0:
            ret_patches = mpatch_i
            ret_labels  = mpatch_m
            ret_imlabel = np.ones_like(mpatch_m)*i
        else:
            tmp_i =np.ones_like(mpatch_m)*i
            ret_patches = np.concatenate((ret_patches, mpatch_i), axis=0)
            ret_labels  = np.concatenate((ret_labels, mpatch_m), axis=0)
            ret_imlabel = np.concatenate((ret_imlabel, tmp_i), axis=0)

    print("Done              ", flush=True)

    if mimage_dir is not None:
        return ret_patches, ret_labels, ret_imlabel, ret_mpatches, ret_names

    return ret_patches, ret_labels, ret_imlabel
