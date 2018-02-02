import cv2
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from shapely.geometry import shape
from sklearn.cluster import MeanShift, estimate_bandwidth

MAX_SIZE=1000

def adjust_contrast(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    newImage = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return newImage

def scale(img):
    height, width = img.shape[:2]
    if(height<=MAX_SIZE and width<=MAX_SIZE):
        return img
    scale = MAX_SIZE/max(height,width)
    newImage = cv2.resize(img,(0,0),fx=scale,fy=scale)
    return newImage

def preprocess(img):
    newImage=cv2.GaussianBlur(img,(3,3),0)
    newImage=adjust_contrast(img)
    return newImage


MIN_MATCH_COUNT = 10

img = cv2.imread('Images/10_2.jpg')    #queryImage
img=scale(img)
img2 = preprocess(img)
patternsPaths=[ ('10','Images/Patterns/10.jpg'),
                ('100','Images/Patterns/100.jpg'),
                ('50','Images/Patterns/50_3.jpg'),
                ('20','Images/Patterns/20.jpg'),
                ('200','Images/Patterns/200.jpg')]

patternsImages=[]
for key,value in patternsPaths:
    patternsImages.append((key,cv2.imread(value,0)))

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()


# find the keypoints and descriptors with SIFT
patternsKeys=[]
for key,value in patternsImages:
    kp,des=sift.detectAndCompute(value,None)
    patternsKeys.append((key,kp,des))
kp2, des2 = sift.detectAndCompute(img2,None)


x = np.array([kp2[0].pt])
for i in range(len(kp2)):
    x = np.append(x, [kp2[i].pt], axis=0)
x = x[1:len(x)]

bandwidth = estimate_bandwidth(x, quantile=0.05, n_samples=105)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
ms.fit(x)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

s = [None] * n_clusters_
for i in range(n_clusters_):
    l = ms.labels_
    d, = np.where(l == i)
    print(d.__len__())
    s[i] = list(kp2[xx] for xx in d)

des2_ = des2

polygons=[]
try:
    for j,(key,kp,des) in enumerate(patternsKeys):
        img1=patternsImages[j][1]
        kp1=kp
        des1=np.float32(des)
        for i in range(n_clusters_):

            kp2 = s[i]
            l = ms.labels_
            d, = np.where(l == i)
            des2 = des2_[d, ]

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            des2 = np.float32(des2)

            matches = flann.knnMatch(des1, des2, 2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

            if len(good)>3:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)

                if M is None:
                    print ("No Homography")
                else:
                    matchesMask = mask.ravel().tolist()

                    h,w = img1.shape
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts,M)

                    pt=np.int32(dst)
                    intersects=False
                    polygonCurrent=Polygon([(pt[0][0][0],pt[0][0][1]),(pt[1][0][0],pt[1][0][1]),(pt[2][0][0],pt[2][0][1]),(pt[3][0][0],pt[3][0][1])])
                    for key2,polygon in polygons:
                        if(polygonCurrent.intersects(polygon)):
                            intersects=True
                            break
                    if(intersects):
                        continue

                    points = np.int32(np.asarray(polygonCurrent.exterior.coords))
                    valid=True
                    for point in points:
                        if (point[0] < 0 or point[1]<0 or point[0]>img2.shape[1] or point[1]>img2.shape[0]):
                            valid = False
                            break
                    if valid==False:
                        continue
                    if(shape(polygonCurrent).area)>50:
                        polygons.append((key,polygonCurrent))
            else:
                print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
                matchesMask = None
except:
    print('error')
sum=0 #final sum

for key,polygon in polygons:
    points = np.int32(np.asarray(polygon.exterior.coords))
    img = cv2.polylines(img, [points], True, 0, 3, cv2.LINE_AA)
    cv2.putText(img,key,(points[0][0],points[0][1]),cv2.FONT_HERSHEY_SIMPLEX,1,0,3,cv2.LINE_AA)
    sum+=int(key)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Suma: {}".format(sum)), plt.show()