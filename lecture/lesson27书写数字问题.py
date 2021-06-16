import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   #目的:不显示多余提示只显示错误信息

import tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

assert tf.__version__.startswith('2.')

def preprocess(x, y): #数据预处理

    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x,y


(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data() #
print(x.shape, y.shape)


batchsz = 128

db = tf.data.Dataset.from_tensor_slices((x,y))
#print(db)
db = db.map(preprocess).shuffle(10000).batch(batchsz)
#print(db)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(batchsz)

db_iter = iter(db)
sample = next(db_iter)
print('batch:', sample[0].shape, sample[1].shape)  #此3行代码只是为了查看数据，不需用写


model = Sequential([
    layers.Dense(256, activation=tf.nn.relu), # [b, 784] => [b, 256]
    layers.Dense(128, activation=tf.nn.relu), # [b, 256] => [b, 128]
    layers.Dense(64, activation=tf.nn.relu), # [b, 128] => [b, 64]
    layers.Dense(32, activation=tf.nn.relu), # [b, 64] => [b, 32]
    layers.Dense(10) # [b, 32] => [b, 10], 330 = 32*10 + 10
])
model.build(input_shape=[None, 28*28])
model.summary()  #可以查看神经网络的结构
# w = w - lr*grad
optimizer = optimizers.Adam(lr=1e-3)

def main():


    for epoch in range(10):


        for step, (x,y) in enumerate(db):

            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape() as tape:
                # [b, 784] => [b, 10]
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                # [b]
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss_ce = tf.reduce_mean(loss_ce) #将loss_ce变成标量，此处有点不清楚

            grads = tape.gradient(loss_ce, model.trainable_variables)
            #model.trainable_variables表示所有的w和b
            optimizer.apply_gradients(zip(grads, model.trainable_variables)) #参数更新语法
            #zip(a,b)表示按a*b的顺序相乘


            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss_ce), float(loss_mse))


        # test
        total_correct = 0
        total_num = 0
        for x,y in db_test:

            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])
            # [b, 10]
            logits = model(x)
            # logits => prob, [b, 10]
            prob = tf.nn.softmax(logits, axis=1)  #将结果转换成probability概率;softmax可以使输出为（0，1）的值，且和为1
            # [b, 10] => [b], int64
            pred = tf.argmax(prob, axis=1) #查找概率最大的所在处，并变成一维数组
            pred = tf.cast(pred, dtype=tf.int32) # 不清楚为什么要把int64转换成int32
            # pred:[b]
            # y: [b]
            # correct: [b], True: equal, False: not equal
            correct = tf.equal(pred, y) #判断b张图片中哪些分类是正确的的，是布尔数据类型
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32)) #将布尔数据转换成数字，并求和b张图片中正确分类的个数

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)







if __name__ == '__main__':
    main()
    #当你要导入某个模块，但又不想该模块的部分代码被直接执行，那就可以这一部分代码放在“if __name__=='__main__':”内部。
    #https://www.zhihu.com/question/49136398
    #https://blog.konghy.cn/2017/04/24/python-entry-program/