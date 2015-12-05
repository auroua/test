import tensorflow as tf

# state = tf.Variable(0, name='counter')
#
# one = tf.constant(1)
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)
#
# init_op = tf.initialize_all_variables()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     print sess.run(state)
#     for _ in range(3):
#         sess.run(update)
#         print sess.run(state)

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    print sess.run([output], feed_dict={input1: [7.], input2: [2.]})