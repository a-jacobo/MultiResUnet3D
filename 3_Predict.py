import numpy as np
import os
from MultiResUNet3D import MultiResUnet3D
from tensorflow.keras import backend as K
from skimage import io
from skimage.exposure import rescale_intensity
from os.path import join
from math import ceil, log


def calc_padding(edge_coords,img_size):
    '''
    Calculates how much of the patch sit inside the original image
    and how much padding outside the image is needed

    Arguments: 
        edge_coords: coordinates of the edges of the patch (including padding)
        img_size : size of the full image
    
    Returns:
        pad: Ammount of padding that lands outside of the bounds of the original image
             and needs to be padded symmetrically.
        new_edge_coords: fraction of the patch coordinates that fit inside the image.
    '''
    pad = np.zeros((3,2),dtype=np.int)
    new_edge_coords = np.copy(edge_coords)
    for j in range(3):
        if  edge_coords[j,0] < 0:
            pad[j,0] = -edge_coords[j,0]
            new_edge_coords[j,0] = 0 
        if img_size[j] - edge_coords[j,1] < 0:
            pad[j,1] = (edge_coords[j,1] - img_size[j])
            new_edge_coords[j,1] = img_size[j] 
    return pad,new_edge_coords

batchSize = 8

data_path = 'C:/Users/ajaco/Work/Cell_Segmentation/CARE/CARE_Restoration/segmentation_cell_body_nucleus/one_channel/data/predict'
filename = 'myo6b_bactn_gfp_2dpf_w1iSIM488-525_s1_t30.TIF'

image = rescale_intensity(io.imread(join(data_path,filename)).astype(np.float32),in_range='image',out_range=(0,1.))

sz,sy,sx = np.array(image.shape,dtype = np.int)

n_tiles = np.array([1,16,16])

block_size = np.ceil((np.array([sz,sy,sx])/n_tiles)).astype(np.int) 
pad_size = np.array([[32,33],[16,16],[16,16]],dtype=np.int) # Left and right padding in z,y,x directions
ps_z,ps_y,ps_x = block_size+np.sum(pad_size,axis=1) # patch size

z_begs=np.arange(0,sz,block_size[0])
y_begs=np.arange(0,sy,block_size[1])
x_begs=np.arange(0,sy,block_size[2])

img_patches = []

# Get padded patches from the image, and add symmetric padding at the edges

for z in z_begs:
    for y in y_begs:
        for x in x_begs:
            edge_coords=np.array([[z-pad_size[0,0],z+block_size[0]+pad_size[0,1]],
                                  [y-pad_size[1,0],y+block_size[1]+pad_size[1,1]],
                                  [x-pad_size[2,0],x+block_size[2]+pad_size[2,1]]],dtype=np.int)
            padding, edge_coords = calc_padding(edge_coords,(sz,sy,sx))
            patch = image[edge_coords[0,0]:edge_coords[0,1],
                          edge_coords[1,0]:edge_coords[1,1],
                          edge_coords[2,0]:edge_coords[2,1]]
            if padding.any():
                patch = np.pad(patch,padding,mode='symmetric')
            img_patches.append(patch)

img_patches = np.array(img_patches)

img_patches = img_patches[:,:,:,:,np.newaxis]

# Create newtwork and load weights
model = MultiResUnet3D(ps_z,ps_y,ps_x,1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("weights.hdf5")

# Predict labels
mask_patches = model.predict(x=img_patches, batch_size=batchSize, verbose=1)

# Crop label tiles to remove padding
mask_patches = (255*(mask_patches[:,pad_size[0,0]:pad_size[0,0]+block_size[0],
                                   pad_size[1,0]:pad_size[1,0]+block_size[1],
                                   pad_size[2,0]:pad_size[2,0]+block_size[2],0]) ).astype(np.int8)

mask = np.empty(block_size*n_tiles,dtype=np.int8)

# Assemble the image from the tiles 
j=0
for z in z_begs:
    for y in y_begs:
        for x in x_begs:
            mask[z:z+block_size[0]
                ,y:y+block_size[1],
                 x:x+block_size[2]] = mask_patches[j,:,:,:]
            j+=1

io.imsave(filename[:-4]+'_label.TIF',mask,check_contrast=False)