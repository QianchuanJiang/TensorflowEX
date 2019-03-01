import numpy as np
import pickle
import cv2 as cv
# 读取并载入训练数据集的方法；
def unpickle(filename):
    with open(filename, 'rb') as fo:
        d = pickle.load(fo, encoding='latin1')
        return d
test = unpickle('cifar10-dataset/test_batch')
X_test = test['data'][:5000, :]
y_test = test['labels'][:5000]
print(X_test.shape)
labelNames = {0: 'airplane',
              1: 'automobile',
              2: 'bird',
              3: 'cat',
              4: 'deer',
              5: 'dog',
              6: 'frog',
              7: 'horse',
              8: 'ship',
              9: 'truck',
}
for i in range(5000):
    Image = X_test[i, :]
    Image = np.reshape(Image, [3, 32, 32])
    imgRGB = np.ones([32, 32, 3], np.uint8)
    for j in range(3):
        img = Image[j]
        img = np.reshape(img, [32, 32])
        imgRGB[:, :, j] = img
    # print(img)
    # cv.imshow("dddd", img)
    cv.imwrite('Images/image'+str(i)+'class_' + labelNames.get(y_test[i]) + '.jpg', imgRGB)
cv.waitKey(0)