import numpy as np
import os
from tqdm import tqdm
from skimage.io import imread
from skimage.exposure import rescale_intensity
from os.path import join
from sklearn.model_selection import train_test_split


patch_size = (16,64,64)
hs_patch_z = np.int(patch_size[0]/2) # Half of the patch size in the z dimmension
hs_patch_y = np.int(patch_size[1]/2) # Half of the patch size in the y dimmension
hs_patch_x = np.int(patch_size[2]/2) # Half of the patch size in the x dimmension

n_patches_per_image = 256

data_path = 'C:/Users/ajaco/Work/Cell_Segmentation/CARE/CARE_Restoration/segmentation_cell_body_nucleus/one_channel/data/train/low'
mask_path = 'C:/Users/ajaco/Work/Cell_Segmentation/CARE/CARE_Restoration/segmentation_cell_body_nucleus/one_channel/data/train/GT'

# Retreive image file names
data_files = [i for i in os.listdir(data_path) if '.TIF' in i]
mask_files = [i for i in os.listdir(mask_path) if '.TIF' in i]

# Find if for every image there is a mask file

missing_files = [i for i in data_files if i not in mask_files]

if len(missing_files)!=0:
    print('[ERROR] The following mask files are missing:')
    print(missing_files)
    exit()
else:
    print('[INFO] Found {} images'.format(len(data_files)))

print('[INFO] Retreiving image patches')

X = []
Y = []
for file in tqdm(data_files):
    # Load image and corresponding mask data and normalize both to [0,1]
    image = rescale_intensity(imread(join(data_path,file)).astype(np.float32),in_range='image',out_range=(0,1.))
    mask = imread(join(mask_path,file)) > 0
    mask = rescale_intensity(mask.astype(np.float32),in_range='image',out_range=(0,1.))
    sz,sy,sx = image.shape
    for i in range(n_patches_per_image):
        # Get random center coordinates for the patch
        center_z = np.random.randint(hs_patch_z,sz-hs_patch_z +1)
        center_y = np.random.randint(hs_patch_y,sy-hs_patch_y +1)
        center_x = np.random.randint(hs_patch_x,sx-hs_patch_x +1)
        
        # Extract patch from the image
        img_patch = image[center_z-hs_patch_z:center_z+hs_patch_z,
                    center_y-hs_patch_y:center_y+hs_patch_y,
                    center_x-hs_patch_x:center_x+hs_patch_x,]
        
        # Extract patch from the mask
        mask_patch = mask[center_z-hs_patch_z:center_z+hs_patch_z,
                    center_y-hs_patch_y:center_y+hs_patch_y,
                    center_x-hs_patch_x:center_x+hs_patch_x,]
        X.append(img_patch)
        Y.append(mask_patch)

X=np.array(X)
Y=np.array(Y)
X = X[:,:,:,:,np.newaxis]
Y = Y[:,:,:,:,np.newaxis]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print('[INFO] Train set shapes are:')
print(X_train.shape)
print(Y_train.shape)
print('[INFO] Test set shapes are:')
print(X_test.shape)
print(Y_test.shape)
np.savez('training_data.npz',X_train=X_train, Y_train=Y_train, X_test =X_test, Y_test=Y_test)
print('[INFO] Data saved to training_data.npz')