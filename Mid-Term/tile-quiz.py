import cv2
import os
import argparse
import numpy as np

def read_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
    images = [cv2.imread(os.path.join(folder_path, img)) for img in image_files]
    return images

def stitch_images(images):
    #
    # Add your solution here
    #
    pass

def combine_images_into_grid(images, rows=2, cols=2):
    if not images:
        print("No images found in the directory.")
        return None

    # Resize images to a standard size for uniform display
    max_height, max_width = 150, 150
    resized_images = [cv2.resize(img, (max_width, max_height)) for img in images]

    # Create blank images to complete the grid if necessary
    blank_image = np.zeros_like(resized_images[0])
    while len(resized_images) < rows * cols:
        resized_images.append(blank_image)

    # Arrange images into rows and columns
    rows_of_images = [np.hstack(resized_images[i * cols:(i + 1) * cols]) for i in range(rows)]
    combined_image = np.vstack(rows_of_images)

    return combined_image

def display_stitched_image(image):
    if image is None:
        return

    cv2.imshow('Stitched Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stitch 9 images and display.')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Path to the folder containing images')
    args = parser.parse_args()

    # Read images from the given folder
    images = read_images_from_folder(args.directory)

    # Once you start, ignore combine_images_into_grid()
    # Instead, complete stitch_image(), if needed you can add more arguments
    # at stitch_image()
    stitched_image = combine_images_into_grid(images, rows=2, cols=2)
    # stitched_image = stitch_image(images) 

    # Display the stitched image
    display_stitched_image(stitched_image)

if __name__ == '__main__':
    main()
