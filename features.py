import base64
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

path =  'train_frames/agzpasxmwv.mp4/'

def feature_extraction(path, name):
    text_path = '/texts' + name
    img = imread(path)
    #imshow(img)
    #print(img.shape)
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True) 
    #ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray) 
    string = base64.b64encode(hog_image)
    file = open(text_path, "w+")
    file.write(string)

