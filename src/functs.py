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

    line = f"{N} {int(imshape[1])} {int(imshape[0])}\n"
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
    ret = xvec.copy()
    return (ret - xvec.min(0))/(fac + 0.001)


def creat_marker_label_image(gt_img, markers, mlabel, bin_label=True):
    ret_img = np.zeros_like(gt_img)
    if bin_label:
        for m,l in zip(markers, mlabel):
            if l==1:
                ret_img[m[0],m[1]]=2
            else:
                ret_img[m[0],m[1]]=1
    else: # for mnist and others
        for m,l in zip(markers, mlabel):
                ret_img[m[0],m[1]]=l+1
    return ret_img



def new_marker_is_distance_from_set(new_marker, markers, marker_size, ratio=1.0):

    newm = np.array(new_marker)
    
    for m in markers:
        if np.linalg.norm(np.array(m) - newm , ord=2) < marker_size * ratio:
            return False

    return True


def cresce_disco(new_marker, marker_size):

    disc = []
    for yi in range(new_marker[0]-marker_size, new_marker[0]+marker_size +1):
        for xi in range(new_marker[1]-marker_size, new_marker[1]+marker_size +1):
            if not new_marker_is_distance_from_set(new_marker, [[yi,xi]], marker_size, ratio=1.0):
                disc.append([yi,xi])

    return disc
            

def gen_random_disk_markers(image, gt, num_markers, marker_size):
    ysize=image.shape[0]
    xsize=image.shape[1]

    Nimg = ysize*xsize
    
    markers_centers = []
    markers_indexes = []


    # gera os centros dos marcadores
    while len(markers_centers) < num_markers:
        index = np.random.randint(0, high=Nimg-1)
        yi = index%(ysize-1)
        xi = int(index/(ysize-1))

        if index in markers_indexes:
            continue

        # verifica se há uma distancia com outros marcadores
        if not new_marker_is_distance_from_set([yi,xi], markers_centers, marker_size, ratio=3.5):
            continue

        # verifica se há uma distancia com as bordas
        # if not new_marker_is_distance_from_set([yi,xi], [[0,0],[ysize-1,0], [0,xsize-1], [ysize-1, xsize-1]], marker_size, ratio=10.0):
        if yi >= ysize-3 or yi < 3 or xi >= xsize-3 or xi < 3:
            continue

        markers_indexes.append(index)
        markers_centers.append([yi,xi])


    # cresce os marcadores
    markers = []
    for m in markers_centers:
        adj_points = cresce_disco(m, marker_size)

        markers = markers + adj_points
    
    mlabel = []
    
    for m in markers:
        mlabel.append(gt)
    
    return markers, mlabel

