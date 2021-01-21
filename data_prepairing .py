import os
from PIL import Image

num_files_desired = 1000

folder_name = 'datasets/green_olive'
COUNT = 1

# Function to increment count
# to make the files sorted.
def increment():
    global COUNT
    COUNT = COUNT + 1

# Function to rename images
# to make files with similar name
def rename_all_files_in_directory():
    os.chdir('./' + folder_name)
    path = os.getcwd()
    print(os.getcwd())
    for f in os.listdir():
        f_name, f_ext = os.path.splitext(f)
        f_name = "green_olive_" + str(COUNT)
        increment()

        new_name = '{}{}'.format(f_name, f_ext)
        os.rename(f, new_name)
    resize_all_photo(path)

# Function to resize images
# makes the all files one size
def resize_all_photo(path):
    folder_path = path + '/'
    for item in os.listdir(folder_path):
        if os.path.isfile(folder_path + item):
            im = Image.open(folder_path + item)
            f, e = os.path.splitext(folder_path + item)
            imResize = im.resize((416,416), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)

if __name__ == '__main__':
    rename_all_files_in_directory()