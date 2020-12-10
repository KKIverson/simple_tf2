import tensorflow as tf
# tf2 默认使用eager execution，若需关闭，则：tf.compat.v1.disable_eager_execution()

# 定义一个随机数（标量）
random_float = tf.random.uniform(shape=())

# 定义一个有2个元素的零向量
zeros = tf.zeros(shape=2)

# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])
