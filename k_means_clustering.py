import cv2
from numpy.linalg import norm
import numpy as np

path = 'C:/Users/aky20/source/repos/MAT345_PY/MAT345_PY/aa.jpg'
img = cv2.imread(path)

K = 4
iteration = 100
width,height,z = img.shape
img2D = img.reshape(width*height,z)

np.random.RandomState(123)
random_idx = np.random.permutation(img2D.shape[0])
centroids = img2D[random_idx[:K]]
for i in range(iteration):
    old_centroids = centroids
    distance = np.zeros((img2D.shape[0], K))
    for j in range(K):
        row_norm = norm(img2D-old_centroids[j, :], axis=1)
        distance[:, j] = np.square(row_norm)

    labels = np.argmin(distance, axis=1)

    centroids = np.zeros((K, img2D.shape[1]))
    for k in range(K):
        centroids[k, :] = np.mean(img2D[labels == k, :], axis=0)

    if np.all(old_centroids == centroids):
        break

center = np.uint8(centroids)
res = center[labels.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()