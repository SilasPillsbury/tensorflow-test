import tensorflow as tf

#initialize variables
def run(inde=[1,2,3,4],depe=[2,5,10,17]):
  sess = tf.Session()

  Q = M = tf.Variable([0.4], dtype=tf.float32)
  M = tf.Variable([0.4], dtype=tf.float32)
  W = tf.Variable([0.3], dtype=tf.float32)
  b = tf.Variable([-0.3], dtype=tf.float32)
  x = tf.placeholder(tf.float32)
  y = tf.placeholder(tf.float32)


  #functions and operations
  
  linear_model = Q*x*x*x + M*x*x + W*x + b
  #linear_model = M*tf.square(x) + W*x + b + 5
  squared_deltas = tf.square(linear_model - y)
  loss = tf.reduce_sum(squared_deltas)

  #train/run
  optimizer = tf.train.GradientDescentOptimizer(0.0001)
  train = optimizer.minimize(loss)
  init = tf.global_variables_initializer()
  sess.run(init)
  for i in range(10000):
    sess.run(train, {x: inde, y: depe})

  print("Q,M,W,b:",sess.run([Q,M,W,b]))
  print(sess.run(loss, {x: inde, y: depe}))
  return M, W, b, x, y

run()
