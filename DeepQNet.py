from keras.models import *
from keras.layers import *
from keras.optimizers import *


def DeepQNet(input_dim, output_dim, lr):
    inputs = Input(input_dim)
    dense1 = Dense(128, activation='relu')(inputs)
    dense2 = Dense(128, activation='relu')(dense1)
    dense3 = Dense(128, activation='relu')(dense2)
    dense4 = Dense(output_dim, activation='linear')(dense3)
    model = Model(input=inputs, output=dense4)
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')
    return model


def load_DeepQNet(path):
    return load_model(path)
