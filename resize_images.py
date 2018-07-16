import numpy as np
from tqdm import tqdm
from scipy import misc as cv2
import glob
import tensorflow as tf
from PIL import Image
from skimage import transform
import copy
from collections import Counter
from random import randint
from skimage.io import imsave
from random import shuffle
import warnings
warnings.filterwarnings("error")
import imageio

def generate_training_data(folder):
    r=0
    "Gets images for training and returns training data"
    print("Resizing Images..")
    training_data = []
    bag=[]
    img_ext=[]
    with tqdm(total=len(glob.glob(folder+"/*.jpg"))) as pbar:
        for img in glob.glob(folder+"/*.jpg"):
            temp=[]
            #n=cv2.imread(img)
            n = imageio.imread(img)
            #x=np.array(transform.resize(n,[64,64,3],mode='constant'))
            x=np.resize(n,[64,64,3])
            bag.append(x)
            pbar.update(1)
            r+=1
    return bag


path="./blah/image"
bag=generate_training_data("training_images")
i=0
j=0
print("Saving resized images..")
with tqdm(total=len(bag)) as pbar:
    for img1 in bag:
        try:
            imsave(path+str(i)+'.jpg',img1)
        except UserWarning:
            print("image not saved")
            j+=1
            pass
        i+=1
        pbar.update(1)
print("All images saved and "+str(j)+" not saved")
