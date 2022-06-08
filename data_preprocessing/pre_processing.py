from scipy import ndimage
from skimage import morphology
import numpy as np
from pydicom import dcmread


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image


def resample(image, pixel_spacing):
    new_size = [1, 1]

    x_pixel = float(pixel_spacing[0])
    y_pixel = float(pixel_spacing[1])

    size = np.array([x_pixel, y_pixel])

    image_shape = np.array([image.shape[0], image.shape[1]])

    new_shape = image_shape * size
    new_shape = np.round(new_shape)
    resize_factor = new_shape / image_shape

    resampled_image = ndimage.zoom(image, resize_factor)

    return resampled_image


def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image


def crop_image(image, display=False):
    # Create a mask with the background pixels
    mask = image == 0

    # Find the brain area
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    # Remove the background
    croped_image = image[top_left[0]:bottom_right[0],
                         top_left[1]:bottom_right[1]]

    return croped_image


def add_pad(image, new_height=275, new_width=275):
    height, width = image.shape

    final_image = np.zeros((new_height, new_width))

    pad_left = int((new_width - width) // 2)
    pad_top = int((new_height - height) // 2)

    # Replace the pixels with the image's pixels
    final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image

    return final_image


def remove_noise(image):
    segmentation = morphology.dilation(image, np.ones(
        (5,
         5)))  #dilation is used to fill with with given array when 1 is found
    segmentation[
        segmentation <
        0] = 0  # added later else mask appears empty sometime as pixel value comes negative

    labels, label_nb = ndimage.label(
        segmentation
    )  # label_nb are no of features and labels become array with feature indexes

    label_count = np.bincount(
        labels.ravel().astype(np.int64)
    )  #bincount counts frequency of all pos no length=max element+1,  ravel() functions returns contiguous flattened array(1D array with all the input-array elements and with the same type as it
    label_count[
        0] = 0  #    # We don't use the first class since it's the background

    label_count.argmax()
    mask = labels == label_count.argmax()
    mask = ndimage.binary_fill_holes(
        mask)  #it fills holes in the mask by giving it max value
    mask = morphology.dilation(mask, np.ones((3, 3)))

    # Only need this image to filter while training
    # if (mask.sum() < 14644):
    #     raise ValueError('Image too small to analyze')

    # masked_image = mask * image  #can't simply multiplya as window min value is less than 0
    masked_image = np.where(mask == 1, image - image.min(), 0)

    return masked_image


def preprocess(a, b, L, W, ds):
    sampled = resample(a, b)
    hu_image = transform_to_hu(ds, sampled)
    win_image = window_image(hu_image, L, W)
    noise_removed = remove_noise(win_image)
    cropped_image = crop_image(noise_removed)
    padded_image = add_pad(cropped_image)
    return padded_image
