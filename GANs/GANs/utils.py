import numpy as np

def prepro_dataset(images):
    """
    """
    images = (images - np.mean(images)) / np.std(images)
    # images = images / np.max(images)
    images = images.astype('float32')
    return images
