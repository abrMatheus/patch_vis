import numpy as np

import PIL
from PIL import Image
from skimage.measure import regionprops

def read_image(path):
    return np.array(Image.open(path))

def save_image(arr, path):
    img = Image.fromarray(arr)
    img.save(path)


def save_marker(markers, mlabel, path, imshape):
    N = int(len(markers))


    f = open(path, "w")

    line = f"{N} {int(imshape[0])} {int(imshape[1])}\n"
    f.writelines([line])

    for m,l in zip(markers, mlabel):
        line = f"{int(m[1])} {int(m[0])} -1 {int(l)} 0\n"  #x y -1 label 0
        f.writelines([line])
    f.close()
    

def read_markers(mpath):

    markers = []
    labels  = []

    with open(mpath, 'r') as f:
        lines = f.readlines()
        line = lines[0]
        # print("line0", line.strip().split(" "))
        for line in lines[1:]:
            nline = line.strip().split(" ")
            # print()
            markers.append([int(nline[1]), int(nline[0])])
            labels.append(int(nline[3]))
            # break
    
    return markers, labels



# TODO: create a function for generate supervoxels image based on different methods SLIC for example.


def gen_svox_center(svox_img, gt_image):
    regions = regionprops(svox_img)

    gt_image=gt_image/gt_image.max()
    markers = []
    mlabel  = []
    for props in regions:
        cy, cx = props.centroid  # Centroid is (row, column)
        cy, cx = int(cy), int(cx)
        # print("cy, cx", cy,cx)
        markers.append([cy,cx])

        mlabel.append(gt_image[cy,cx])

    return markers, mlabel


def min_max_norm(xvec):
    fac = (xvec.max(0) - xvec.min(0))
    return (xvec - xvec.min(0))/(fac + 0.001)


def creat_marker_label_image(gt_img, markers, mlabel):
    ret_img = np.zeros_like(gt_img)
    for m,l in zip(markers, mlabel):
        if l==1:
            ret_img[m[0],m[1]]=2
        else:
            ret_img[m[0],m[1]]=1
    return ret_img


