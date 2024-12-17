import os
import cv2
from LoFTR import LoFTR_Matcher

images = []
for dir, _, files in os.walk('data/prepared_images'):
    for file in files:
        images.append(cv2.imread(os.path.join(dir, file)))

print(images)

# For best matching performance, an input image size should be a multiple of IMAGE_SIZE
IMAGE_SIZE = (1098, 1098)
matcher = LoFTR_Matcher(IMAGE_SIZE)

def outputs_matches(image0, image1, conf=0.8, accurate=False):
    corresp = matcher(image0, image1, conf, accurate)

    # Inliers check
    if corresp['inliers'] is None:
        print("No inliers found. Skipping these images.")
        return

    # If there are inliers, count the quantity and print the result
    keypoints_num = len(corresp['inliers'])
    inliers_num = sum(corresp['inliers'])[0]
    ratio = inliers_num / float(keypoints_num)

    print(f'Key points: {keypoints_num}, Inlier points: {inliers_num}, Ratio: {ratio:.2f}')

    # Visualization
    matcher.draw_matches(corresp)

    img_ids = [0, 6, 16, 18]

    for i in range(len(img_ids)):
        for j in range(i + 1, len(img_ids)):
            idx1 = img_ids[i]
            idx2 = img_ids[j]
            print(f"Processing images {idx1} and {idx2}...")
            outputs_matches(images[idx1], images[idx2], conf=0.9)

