from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Conv1D,Conv2D,Dropout,concatenate,Reshape,Flatten
from tensorflow.python.keras.layers import CuDNNLSTM, LSTM
from tensorflow.keras.activations import tanh

def MCLmodel(input_shape1=[4,12,1], input_shape2=[12,1]):
    input1 =Input(input_shape1)
    input2 =Input(input_shape2)
    input3=Input(input_shape2)
    input4=Input(input_shape2)
    input5=Input(input_shape2)
    
    #Cnvolutional Block
    x1=Conv2D(5,(2,3),padding='same', activation="relu", kernel_initializer='glorot_uniform')(input1)
    x2=Conv1D(5,3,padding='causal', activation="relu", kernel_initializer='glorot_uniform')(input2)
    x2_reshape=Reshape([-1,12,5])(x2)
    x3=Conv1D(5,3,padding='causal', activation="relu", kernel_initializer='glorot_uniform')(input3)
    x3_reshape=Reshape([-1,12,5])(x3)
    x4=Conv1D(5,3,padding='causal', activation="relu", kernel_initializer='glorot_uniform')(input4)
    x4_reshape=Reshape([-1,12,5])(x4)
    x5=Conv1D(5,3,padding='causal', activation="relu", kernel_initializer='glorot_uniform')(input5)
    x5_reshape=Reshape([-1,12,5])(x5)
    x=concatenate([x2_reshape,x3_reshape,x4_reshape,x5_reshape],axis=1)
    x=Conv2D(5,(2,3),padding='same', activation="relu", kernel_initializer='glorot_uniform')(x)
    x=concatenate([x1,x])
    x=Conv2D(10,(4,2),padding='valid', activation="relu", kernel_initializer='glorot_uniform')(x)
    
    #LSTM Unit
    x = Reshape(target_shape=((11,10)))(x)
    x = CuDNNLSTM(units=8)(x)
    
    #DNN
    x = Dense(16)(x)
    x=tanh(x)

    x = Dense(2)(x)

    model = Model(inputs = [input1,input2,input3,input4,input5],outputs = x)

    return model
