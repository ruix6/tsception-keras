from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate, LeakyReLU


# 定义时域卷积块
def conv_block(input, out_chan, kernel, step, pool):
    x = Conv2D(out_chan, kernel, strides=step, padding='same', use_bias=False)(input)
    x = LeakyReLU()(x)
    x = AveragePooling2D(pool_size=(1, pool), strides=(1, pool))(x)

    return x


def Tsception(num_classes, Chans, Samples, sampling_rate, num_T, num_S, hidden, dropout_rate, pool=8):

    '''
    input_size: 输入数据的维度,(chans, samples, 1)
    '''
    inception_window = [0.5, 0.25, 0.125]
    # 定义输入层
    input = Input(shape=(Chans, Samples, 1))
    # 定义时域卷积层
    x1 = conv_block(input, num_T, (1, int(sampling_rate * inception_window[0])), 1, pool)
    x2 = conv_block(input, num_T, (1, int(sampling_rate * inception_window[1])), 1, pool)
    x3 = conv_block(input, num_T, (1, int(sampling_rate * inception_window[2])), 1, pool)
    # 在height维度上进行拼接
    x = concatenate([x1, x2, x3], axis=2)
    x = BatchNormalization()(x)
    # 定义空域卷积层
    y1 = conv_block(x, num_S, (Chans, 1), (Chans, 1), int(pool*0.25))
    y2 = conv_block(x, num_S, (int(Chans*0.5), 1), (int(Chans*0.5), 1), int(pool*0.25))
    # 在width维度上进行拼接
    y = concatenate([y1, y2], axis=1)
    y = BatchNormalization()(y)
    # 定义fusion_layer
    z = conv_block(y, num_S, (3, 1), (3, 1), 4)
    z = BatchNormalization()(z)
    # 定义全局平均池化层
    z = AveragePooling2D(pool_size=(1, z.shape[2]))(z)
    z = Flatten()(z)
    # 全连接层
    z = Dense(hidden, activation='relu', use_bias=False)(z)
    z = Dropout(dropout_rate)(z)
    z = Dense(num_classes, activation='softmax', use_bias=False)(z)

    return Model(inputs=input, outputs=z)


if __name__ == '__main__': 
    model = Tsception(num_classes=2, Chans=28, Samples=512, sampling_rate=128, num_T=64, num_S=32, hidden=32, dropout_rate=0.5)
    model.summary()