import numpy as np
import pickle
import cv2 as cv
import random
import matplotlib.pyplot as plt
# 读取并载入训练数据集的方法；
def unpickle(filename):
    with open(filename, 'rb') as fo:
        d = pickle.load(fo, encoding='latin1')
        return d

'''
导入全部CIFAR10训练集的方法；
'''
def RaadTrianData():
    data1 = unpickle('cifar10-dataset/data_batch_1')
    data2 = unpickle('cifar10-dataset/data_batch_2')
    data3 = unpickle('cifar10-dataset/data_batch_3')
    data4 = unpickle('cifar10-dataset/data_batch_4')
    data5 = unpickle('cifar10-dataset/data_batch_5')
    #  把训练集整合到一个数组中；
    X_train = np.concatenate((data1['data'], data2['data'], data3['data'], data4['data'], data5['data']), axis=0)
    # print("ffffff" + str(X_train.shape))
    # 把训练集的标签也整合到一个数组中；
    y_train = np.concatenate((data1['labels'], data2['labels'], data3['labels'], data4['labels'], data5['labels']),axis=0)
    # print("ffffff" + str(y_train.shape))
    return X_train, y_train


test = unpickle('cifar10-dataset/test_batch')
X_test = test['data'][:5000, :]
# print(X_test.shape)
y_test = test['labels'][:5000]
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

# 读取图片并转化成卷积层读取数据的个格式；
def ReadImage(Img):
    # 导入图片；
    # Img = cv.imread('Images/image0class_cat.png', cv.IMREAD_COLOR)
    # 按照颜色通道分成三个数组；
    r, g, b = cv.split(Img)
    ImgData = np.ones([3, 32*32], np.uint8)
    # 整合成一个颜色数组；
    ReadImg = cv.merge([r, g, b])
    for i in range(3):
        ImDataR = ReadImg[:, :, i]
        ImDataR = np.reshape(ImDataR, [32*32])
        ImgData[i, :] = ImDataR
    ImgData = np.reshape(ImgData, [32*32*3])
    # print(ImgData)
    # print(ImgData.shape)
    return ImgData

'''
转换成卷积神经网络需要的224*224的3通道图片训练集数据；
Xtrain:导入原始24*24*3的图片;
返回值 TestImageData：[ImageLen,24*24*3]的图片数组，直接用于训练；
'''
def ReadImageData(Xtrain, SaveImage=False):
    ImageLen = len(Xtrain)
    TestImageData = np.ones([ImageLen, 224 * 224 * 3])
    for i in range(ImageLen):
        Image = Xtrain[i, :]
        Image = np.reshape(Image, [3, 32, 32])
        imgRGB = np.ones([32, 32, 3], np.uint8)
        for j in range(3):
            img = Image[j]
            img = np.reshape(img, [32, 32])
            imgRGB[:, :, j] = img
        imgRGB = cv.resize(imgRGB, (224, 224))
        # print(imgRGB)
        if SaveImage == True:
            cv.imwrite('Images/Xtrain' + str(i) + '.png', imgRGB)
        ReadImage(imgRGB)
        # print(ReadImage(imgRGB))
        TestImageData[i, :] = ReadImage(imgRGB)
    # print(TestImageData)
    # print(TestImageData.shape)
    return TestImageData
    #np.save('Test', TestImageData)

# 导入图片集，随机翻转图片；
def RotateImageFunc(imageList):
    # 传入特征数据集；
    # 新建一个数据集格式的数组（:, 3072）大小，用来存储旋转后的图片数据；
    rotateImg = np.ones([len(imageList), 3 * 32 * 32], np.uint8)
    # 遍历每组导入数据；
    for i in range(len(imageList)):
        # 单张图片数据；
        Image = imageList[i, :]
        # 转化成三维形式；
        Image = np.reshape(Image, [3, 32, 32])
        # 定义一个RGB图片格式的数组；
        imgRGB = np.ones([32, 32, 3], np.uint8)
        # 生成图片数据
        for j in range(3):
            img = Image[j]
            img = np.reshape(img, [32, 32])
            imgRGB[:, :, j] = img
        # 定义一个-1到2之间的随机数，来随机不同的翻转参数；
        RotateV = random.randrange(-1, 3)
        # 图片翻转；
        if RotateV == 2:
            rotate = imgRGB
        else:
            rotate = cv.flip(imgRGB, RotateV)
        # imageName = "Images/RotateImage" + str(i) + ".jpg"
        # cv.imwrite(imageName, rotate)
        rotateImg[i, :] = ReadImage(rotate)
    return rotateImg


def DrawImage(a, c):
    plt.plot(a)
    plt.plot(c)
    plt.xlabel('Iter')
    plt.ylabel('Cost')
    plt.title('ACC')
    plt.tight_layout()
    plt.savefig('cnn-tf-cifar10-%s.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    DrawImage([9.85, 12.55, 15.65, 23.688, 25.698, 30.569, 30.985, 29.68545654, 33.56897, 38.546, 40.5896], [2.85, 2.55, 2.65, 2.688, 2.698])
    pass
