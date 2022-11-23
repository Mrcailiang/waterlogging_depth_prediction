import numpy as np
import tensorflow as tf
from data import util
from tensorflow import keras

window_size = 8  # 窗口大小
batch_size = 64  # 训练批次大小
epochs = 200  # 训练epoch
filter_nums = 10  # filter数量
kernel_size = 4  # kernel大小


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

def build_model():
    train_X, train_label, test_X, test_label, std_y, mean_y = get_dataset()
    print(train_X.shape)
    print(train_label.shape)
    model = keras.models.Sequential()
    # model.add(keras.layers.Reshape((8, 2)))
    model.add(tf.keras.layers.GRU(64, activation='relu', return_sequences=True, input_shape=(8, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.GRU(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    model.fit(train_X, train_label, validation_split=0.2, epochs=epochs)
    # 保存模型
    model.save('gru.h5')

if __name__ == '__main__':
    # print(" dsfgadf")
    build_model()
