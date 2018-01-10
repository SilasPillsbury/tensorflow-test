from __future__ import print_function
import tensorflow as tf

sess = tf.Session()
node1 = tf.constant(3.0,dtype=tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
add_and_triple = adder_node*3.0

print(sess.run(add_and_triple, {a : 3, b : 4.5}))


"""
adder_node = node1 + node2
print("sess.run(adder_node):", sess.run(adder_node))
"""

"""
print("node3:",node3)
print(sess.run([node1, node2]))
"""
