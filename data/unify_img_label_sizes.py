import glob
import cv2

unified_img_height = 720
unified_img_width = 1280

# list images/labels whose size is to be unified
images_list = glob.glob('railways_night_5classes/leftImg8bit/train/*.jpg')
labels_list = glob.glob('railways_night_5classes/gtFine/train/*.png')

for img_path in images_list:
    img = cv2.imread(img_path)
    if img.shape != (unified_img_height, unified_img_width, 3):
        # resize img
        img = cv2.resize(img, (unified_img_width, unified_img_height))
        # rewrite resized img into file
        cv2.imwrite(img_path, img)

for label_path in labels_list:
    label = cv2.imread(label_path)
    if label.shape != (unified_img_height, unified_img_width, 3):
        # resize img
        label = cv2.resize(label, (unified_img_width, unified_img_height))
        # rewrite resized img into file
        cv2.imwrite(label_path, label)