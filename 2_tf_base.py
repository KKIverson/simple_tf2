import tensorflow as tf

# tf2 默认使用eager execution，若需关闭，则：tf.compat.v1.disable_eager_execution()

# 定义一个随机数（标量）
random_float = tf.random.uniform(shape=())
# 定义一个有2个元素的零向量
zeros = tf.zeros(shape=2)
# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])
# 查看矩阵A的形状、类型和值
print(A.shape)  # 输出(2, 2)，即矩阵的长和宽均为2
print(A.dtype)  # 输出<type: 'float32'>
print(A.numpy())  # 输出[[1. 2.][3. 4.]]

C = tf.add(A, B)  # 计算矩阵A和B的和
D = tf.matmul(A, B)  # 计算矩阵A和B的乘积
print(C)
print(D)

# 标量求导
x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:  # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)  # 计算y关于x的导数
print(y, y_grad)

# 张量求导
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])
print(L, w_grad, b_grad)



