import tensorflow as tf

#initialize variables
sess = tf.Session()

W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#functions and operations
linear_model = W*x + b
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

#train/run
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
  sess.run(train, {x: [1,2,3,4], y: [0,-1,-2,-3]})

print(sess.run([W,b]))
print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))
