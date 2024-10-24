import cv2
import os
import argparse
import numpy as np

def read_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
    images = [cv2.imread(os.path.join(folder_path, img)) for img in image_files]
    return images

# added code
def find_keypoints_and_match(img1, img2):
    """
    Find keypoints and match between two images using SIFT.
    """
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Use FLANN for feature matching
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    return keypoints1, keypoints2, good_matches

def stitch_images(images):
    #
    # Add your solution here
    #
    if not images:
        print("No images found in the directory.")
        return None
    
    stitched_image = images[0]
    for i in range(1, len(images)):
        keypoints1, keypoints2, good_matches = find_keypoints_and_match(stitched_image, images[i])

        # Extract location of good matches
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        H, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

        # Warp images to create the panorama
        height, width, channels = stitched_image.shape
        next_img = cv2.warpPerspective(images[i], H, (width * 2, height))
        next_img[0:height, 0:width] = stitched_image

        # Crop black borders
        stitched_image = crop_black_borders(next_img)
    
    return stitched_image

    # pass

# added code
def crop_black_borders(image):
    """
    Crop the black borders from the stitched image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y+h, x:x+w]
    return image

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
