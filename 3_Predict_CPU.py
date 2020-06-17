import numpy as np
import os
from MultiResUNet3D import MultiResUnet3D
from tensorflow.keras import backend as K
from skimage import io
from skimage.exposure import rescale_intensity
from os.path import join
from math import ceil, log
from scipy.ndimage.measurements import center_of_mass
from skimage.filters import gaussian


def NextPowerOfTwo(number):
# Returns next power of two following 'number'
    return 2**ceil(log(number,2))

def calc_padding(edge_coords,img_size):
    pad = np.zeros((3,2),dtype=np.int)
    new_edge_coords = np.copy(edge_coords)
    for j in range(3):
        if  edge_coords[j,0] < 0:
            pad[j,0] = -edge_coords[j,0]
            new_edge_coords[j,0] = 0 #np.max(0,edge_coords[j,0]-pad_size[j])
        if img_size[j] - edge_coords[j,1] < 0:
            pad[j,1] = (edge_coords[j,1] - img_size[j])
            new_edge_coords[j,1] = img_size[j] #np.min(edge_coords[j,1],edge_coords[j,0]-pad_size[j])
    return pad,new_edge_coords

batchSize = 1

data_path = 'C:/Users/ajaco/Work/Cell_Segmentation/CARE/CARE_Restoration/segmentation_cell_body_nucleus/one_channel/data/predict'
filename = 'myo6b_bactn_gfp_2dpf_w1iSIM488-525_s1_t30.TIF'

image = rescale_intensity(io.imread(join(data_path,filename)).astype(np.float32),in_range='image',out_range=(0,1.))

orig_sz,orig_sy,orig_sx = np.array(image.shape,dtype = np.int)

sz,sy,sx = np.array(image.shape,dtype = np.int)

mismatch_z = NextPowerOfTwo(sz) - sz #sz % n_tiles[0]
mismatch_y = NextPowerOfTwo(sy) - sy #sy % n_tiles[1]
mismatch_x = NextPowerOfTwo(sx) - sx #sx % n_tiles[2]

print(image.shape)
print(mismatch_z,mismatch_y,mismatch_x)
# If the image is not divisible by the number of tiles, pad it to match.
if (mismatch_z != 0) or (mismatch_y != 0) or (mismatch_x != 0):
    print('[WARNING] Image size is not divisible by number of tiles')
    pdz = (np.floor(mismatch_z/2).astype(int),np.ceil(mismatch_z/2).astype(int))
    pdy = (np.floor(mismatch_y/2).astype(int),np.ceil(mismatch_y/2).astype(int))
    pdx = (np.floor(mismatch_x/2).astype(int),np.ceil(mismatch_x/2).astype(int))
    image = np.pad(image,(pdz,pdy,pdx),mode='symmetric')
    sz,sy,sx = np.array(image.shape,dtype = np.int)

cm = np.array(center_of_mass(gaussian(image))).astype(np.int)
print(cm)

image = image[np.newaxis,:,cm[1]-256:cm[1]+256,cm[2]-256:cm[2]+256,np.newaxis]

p,sz,sy,sx,q = np.array(image.shape,dtype = np.int)

model = MultiResUnet3D(sz,sy,sx,1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("weights.hdf5")

mask = model.predict(x=image, batch_size=batchSize, verbose=1)

#mask = (255*(mask[0,0:orig_sz,0:orig_sy,0:orig_sx,0]>0.5)).astype(np.int8)
#mask = (255*(mask[0,:,:,:,0]>0.5)).astype(np.int8)
mask = (255*(mask[0,:,:,:,0])).astype(np.int8)

io.imsave(filename[:-4]+'_label_CPU.TIF',mask,check_contrast=False)