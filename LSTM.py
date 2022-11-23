import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from math import sqrt
from data import util
from tcn.tcn import TCN
from tensorflow import keras
from keras.models import load_model

window_size = 8  # 窗口大小
batch_size = 64  # 训练批次大小
epochs = 200  # 训练epoch
filter_nums = 10  # filter数量
kernel_size = 4  # kernel大小
train = True


def get_dataset():
    # df = pd.read_csv('./000001_Daily_2006_2018.csv')
    df = util.loadData('data/淮河路新.txt')  # df shape(*,20) (20个值分别为：预测水位、降雨量、风力、雨型、8个过去水位、8个过去雨量)
    df = np.array(df)
    length = len(df)
    np.random.shuffle(df)  # 随机打乱

    # 数据处理
    x_data = df[:, 1:]  # (*,19)
    y_data = df[:, 0].reshape(-1, 1)  # (*,1)

    # 归一化处理
    mean_x = x_data.mean(axis=0).reshape(1, 19)  # axis = 0：压缩行，对各列求均值，返回 1* 19 矩阵
    std_x = x_data.std(axis=0).reshape(1, 19)  # axis = 0,压缩行，对各列求标准差，返回 1* 19 矩阵
    x_data = (x_data - mean_x) / std_x

    mean_y = y_data.mean(axis=0)
    std_y = y_data.std(axis=0)
    y_data = (y_data - mean_y) / std_y

    train_example_num = (int)(length * 0.8)
    # 训练集测试集分割
    train_X = x_data[0:train_example_num, 3:].reshape(-1, 8, 2)
    train_label = y_data[0:train_example_num, :]

    test_X = x_data[train_example_num:, 3:].reshape(-1, 8, 2)
    test_label = y_data[train_example_num:, :]

    return train_X, train_label, test_X, test_label, std_y, mean_y


def get_continueTest():
    data = util.loadData('data/淮河路新.txt')  # df shape(*,20) (20个值分别为：预测水位、降雨量、风力、雨型、8个过去水位、8个过去雨量)
    data = np.array(data)
    # 数据处理
    x_data = data[:, 1:]  # (*,19)
    y_data = data[:, 0].reshape(-1, 1)  # (*,1)
    continueTest = x_data[290:, 3:].reshape(-1, 8, 2)
    continueTestLabel = y_data[290:, :]
    return continueTest, continueTestLabel



def plotTwoModel(pred1,pred2, true):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(range(len(pred1)), pred1, label='pred1')
    plt.plot(range(len(pred2)), pred2, label='pred2')
    plt.plot(range(len(true)), true, label='actual')
    plt.show()

def plotOneModel(pred1, true):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(range(len(pred1)), pred1, label='pred1')
    # plt.plot(range(len(pred2)), pred2, label='pred2')
    plt.plot(range(len(true)), true, label='actual')
    plt.show()

def build_model():
    train_X, train_label, test_X, test_label, std_y, mean_y = get_dataset()
    print(train_X.shape)
    print(train_label.shape)
    if (train == True):
        model = keras.models.Sequential()
        # model.add(keras.layers.Reshape((8, 1)))
        model.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(8, 2)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        model.summary()
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        model.fit(train_X, train_label, validation_split=0.2, epochs=epochs)
        # 保存模型
        model.save('lstm.h5')
        # continueTest, continueTestLabel = get_continueTest()

    if (train == False):
        # 加载模型
        model1 = load_model('tcn-lstm.h5', custom_objects={'TCN': TCN})
        model2 = load_model('tcn.h5', custom_objects={'TCN': TCN})
        model1.compile(optimizer='adam', loss='mae', metrics=['mae'])
        model2.compile(optimizer='adam', loss='mae', metrics=['mae'])

        model1.evaluate(test_X, test_label)
        model2.evaluate(test_X, test_label)
        prediction1 = model1.predict(test_X)
        prediction2 = model2.predict(test_X)
        scaled_prediction1 = prediction1 * std_y + mean_y
        scaled_prediction2 = prediction2 * std_y + mean_y
        scaled_test_label = test_label * std_y + mean_y

        # # 计算mae,rmse,r2
        # absError = []
        # squaredError = []
        # for i in range(len(scaled_prediction)):
        #     error = abs(scaled_prediction[i] - scaled_test_label[i])
        #     absError.append(error)
        #     squaredError.append(error * error)
        #
        # print('mae', sum(absError) / len(absError))
        # print('mse', sum(squaredError) / len(squaredError))
        # print('RMSE ', sqrt(sum(squaredError) / len(squaredError)))
        # print('r-squared', r2_score(scaled_prediction, scaled_test_label))
        plot(scaled_prediction1,scaled_prediction2, scaled_test_label)



if __name__ == '__main__':
    # print(" dsfgadf")
    build_model()
