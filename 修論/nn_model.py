import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   #目的:不显示多余提示只显示错误信息

import numpy as np
import tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import  layers, optimizers, Sequential
import matplotlib.pyplot as plt

x_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')
x_test=np.load('x_test.npy')
y_test=np.load('y_test.npy')
x_train=tf.cast(x_train,dtype=float)
y_train=tf.cast(y_train,dtype=float)
x_test=tf.cast(x_test,dtype=float)
y_test=tf.cast(y_test,dtype=float)
db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(5000)
db_test=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(1000)


model = Sequential([
    layers.Dense(10, activation=tf.nn.tanh),
    layers.Dense(10, activation=tf.nn.tanh),
    layers.Dense(10, activation=tf.nn.tanh),
    layers.Dense(8, activation=tf.nn.tanh),
    layers.Dense(8, activation=tf.nn.tanh),
    layers.Dense(6, activation=tf.nn.tanh),
    layers.Dense(6, activation=tf.nn.tanh),
    layers.Dense(4, activation=tf.nn.tanh),
    layers.Dense(2)
])
model.build(input_shape=[None,6])
model.summary()
optimizer = optimizers.Adam(lr=1e-5)
#optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
def main():
        for epoch in range(30):

            for step, (x,y) in enumerate(db):

                with tf.GradientTape() as tape:
                    tape.watch([x])
                    logits = model(x)
                    loss_mse = tf.reduce_mean(tf.losses.MSE(y, logits))

                grads = tape.gradient(loss_mse, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if step % 100 == 0:
                    print(epoch, step, 'loss:',float(loss_mse))

            #test
            loss_test_mse=np.zeros((10,))
            for step,(x,y) in enumerate(db_test):
                logits=model(x)
                loss_test_mse[step]=float(tf.reduce_mean(tf.losses.MSE(y,logits)))

            loss_test_mse=np.mean(loss_test_mse)
            print("------------------------------------")
            print(epoch,'test_loss:',float(loss_test_mse))
            print("------------------------------------")

model.save('model3.h5')




if __name__ == '__main__':
        main()