import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def data_generator():
    # input image dimensions
    img_rows, img_cols = 28, 28
    #导入手写数据集形状为（28,28）
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #print("11111:**",y_train)
    #转化为（784，2）
    x_train = x_train.reshape(-1, img_rows * img_cols, 1)
    print("222222:",x_train.shape)
    x_test = x_test.reshape(-1, img_rows * img_cols, 1)
    #分类数量为0,1,2,3,4,5,6,7,8,9
    num_classes = 10
    #将标签值转化为one-hot编码
    y_train = to_categorical(y_train, num_classes)
    print("yyyyyy",y_train.shape)
    y_test = to_categorical(y_test, num_classes)

    y_train = np.expand_dims(y_train, axis=2)
    print("zzzzzz", y_train.shape)
    y_test = np.expand_dims(y_test, axis=2)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    print(data_generator())

