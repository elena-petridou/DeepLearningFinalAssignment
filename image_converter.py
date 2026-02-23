'''
Converts all images in a directory to '.npy' format.
Use np.save and np.load to save and load the images.
Use it for training your neural networks in ML/DL projects. 
'''
import os
import glob
from PIL import Image, ImageOps
import numpy as np

class ImageConverter:
    """
    A class to convert images in a specified directory to numpy arrays and save them as .npy files.
    """
    @staticmethod
    def trim_white_space(image: Image.Image) -> Image.Image:
        """
        Trims the white borders from an image.
        
        Args:
            image (PIL.Image.Image): The input image to be trimmed.
        Returns:
            PIL.Image.Image: The trimmed image.
        """
        invert_im = ImageOps.invert(image)
        imageBox = invert_im.getbbox()
        trimmed_image = image.crop(imageBox)
        return trimmed_image

    @staticmethod
    def remove_black_images(images: np.ndarray, 
                            thresh: float = 70) -> np.ndarray:
        """
        Removes images that are predominantly black based on a mean pixel intensity threshold.
        
        Args:
            images (np.ndarray): Array of images to be filtered.
            thresh (float): Threshold for mean pixel intensity below which images are considered black.
        
        Returns:
            np.ndarray: Array of images with predominantly black images removed.
        """
        means = np.mean(images, axis=(1, 2, 3))
        x = images[means > thresh]
        return x
    
    @staticmethod
    def load_dataset(filename: str,
                    path: str,
                    extension: str = "jpg",
                    levels: int = 3,
                    save = False,
                    trim_white = False,
                    remove_black = False,
                    black_thresh: int = 80) -> np.ndarray:
        """
        Iterates through all the flower images in the directory and converts them to RGB values.
        Subsequently, it saves a .npy file of the images, one for train, test, and validation as 
        the images are sorted
        
        Args:
            filename (str): The name of the .npy file to save the images.
            path (str): The directory path where the images are stored.
            levels (int): The number of subdirectory levels to traverse.
            save (bool): Whether to save the converted images as a .npy file.
            trim_white (bool): Whether to trim white borders from the images.
        
        Returns:
            np.ndarray: The npy file containing the convereted images
        """
        x=[]
        path = path + levels*"*/"
        for path_image in glob.glob(path + "*." + extension):
            if os.path.isfile(path_image):
                im = Image.open(path_image).convert("RGB")
                if (trim_white):
                    im = ImageConverter.trim_white_space(im)
                im = im.resize(size=(64,64), 
                               box=None, reducing_gap=None)
                im = np.array(im)
                x.append(im)
        imgset=np.array(x)
        if remove_black:
            imgset = ImageConverter.remove_black_images(imgset, thresh=black_thresh)
        if save:
            np.save(filename,imgset)
        return imgset
