import glob
import cv2 

i = 0
images = glob.glob('C:/Users/ahmet/Desktop/EECE7370/Project/cifar10_64_64/*/*/*.png')
for image in images:
    i = i+ 1
    print(i, ' / ', len(images))

    img = cv2.imread(image)
    img_fin = cv2.resize(img, (64, 64))
    cv2.imwrite(image,img_fin)
    