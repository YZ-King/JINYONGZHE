import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# x: [60k, 28, 28],60k张图片，28*28的像素
# y: [60k]
(x, y), _ = datasets.mnist.load_data()
# x: [0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))   #reduce_min是在tf中查看数据的最小值
print(tf.reduce_min(y), tf.reduce_max(y))


train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)  #对训练集处理，一次训练128个数据点；from_tensor_slices切分传入Tensor的第一个维度，生成相应的dataset。
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)


# [b, 784] => [b, 256] => [b, 128] => [b, 10]
# [dim_in, dim_out], [dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))#方差默认是1，设置成0.1，否则出现梯度爆炸的情况
#truncated_normal裁剪过的正太分布，数据维度是[784,256]，这样x和w1相乘后就能变成256维度的数据，也间接说明第一层有256个神经元
#然后将数据转换成tf.variable类型，使得下面求导过程信息可以被记录
b1 = tf.Variable(tf.zeros([256]))#偏置初始都设置为0，维度为256，使得wx+b可以进行运算
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3

for epoch in range(10): # iterate db for 10
    for step, (x, y) in enumerate(train_db): # for every batch #enumerate()处理可以告诉你当前是那个step #每进行一个batch叫做一个step
        # x:[128, 28, 28]
        # y: [128]

        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28*28])

        with tf.GradientTape() as tape: # 只记录tf.Variable的信息    #涉及到求导部分的计算放在下面
            # x: [b, 28*28]
            # h1 = x@w1 + b1
            # [b, 784]@[784, 256] + [256] => [b, 256] + [256] => [b, 256] + [b, 256]
            h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            ''' broadcast説明
                x = tf.constant([1, 2, 3])
                y = tf.broadcast_to(x, [3, 3])
                print(y)
                tf.Tensor(
                    [[1 2 3]
                     [1 2 3]
                    [1 2 3]], shape = (3, 3), dtype = int32)
            '''
            h1 = tf.nn.relu(h1)
            # [b, 256] => [b, 128]
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            # [b, 128] => [b, 10]
            out = h2@w3 + b3

            # compute loss
            # out: [b, 10]
            # y: [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10) #分类用的，将结果处理一下

            # mse = mean(sum(y-out)^2)
            # [b, 10]
            loss = tf.square(y_onehot - out)
            # mean: scalar
            loss = tf.reduce_mean(loss) #就是求均值

        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # print(grads)
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0]) #保持w1的tf.variable数据类型，进行更新
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])


        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))   #float（）转换成numpy格式
            #总共有60000张图片，每次计算128张图片，大概一共有step=468 次。然后60000张图片重复epoch=10此计算。