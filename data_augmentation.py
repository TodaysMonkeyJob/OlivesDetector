import random
from scipy import ndarray
import os
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

folder_path = 'datasets/green_olive'

# Function to rotate images
# randomly rotate image in any way
def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-180, 180)
    return sk.transform.rotate(image_array, random_degree)

# Function to add noise to images
def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

# Function to flip horizontal images
def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

# Function to augmentate images
# expand our dataset with augmentationed data
def data_augmentation():

    # dictionary of the transformations we defined earlier
    available_transformations = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip
    }

    num_files_desired = 1000

    # find all files paths from the folder
    images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
              os.path.isfile(os.path.join(folder_path, file))]

    num_generated_files = len(images)
    while num_generated_files <= num_files_desired:
        print(num_generated_files)
        # random image from the folder
        image_path = random.choice(images)
        # read image as an two dimensional array of pixels
        image_to_transform = sk.io.imread(image_path)
        # random num of transformation to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))

        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1

        new_file_path = '%s/green_olive_%s.jpg' % (folder_path, num_generated_files)

        # write image to the disk
        io.imsave(new_file_path, transformed_image)
        num_generated_files += 1

if __name__ == '__main__':
    data_augmentation()