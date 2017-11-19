from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from time import time
import pickle
import os

def weight_variable(shape,seed, stddev=0.1):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


# Create the model
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("tau", 2, "Softmax temperature")
tf.app.flags.DEFINE_float("alpha", 0.5, "alpha")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch_size")
tf.app.flags.DEFINE_string("data_dir", "../MNIST_data", "data_dir")
tf.app.flags.DEFINE_string("mode", "baseline", "train mode")
tf.app.flags.DEFINE_string("num", "0","number")
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
global_step = tf.Variable(0, trainable=False)
is_training = tf.placeholder(tf.bool)
v = tf.eye(10)
emb = tf.Variable(v, True)
#emb = tf.Variable(pickle.load(open('mlp_embed.pkl','rb')), False)
x_image = x

#params
W_fc1 = weight_variable([784, 500],10)
b_fc1 = bias_variable([500])
W_fc2 = weight_variable([500, 500],6)
b_fc2 = bias_variable([500])
W_out1 = weight_variable([500, 10], 7,stddev=0.01)
b_out1 = bias_variable([10])
W_out2 = weight_variable([500, 10], 0,stddev=0.01)
b_out2 = bias_variable([10])


#fc1
h_fc1 = tf.matmul(x_image, W_fc1)+ b_fc1 #50*500
a_fc1 = tf.nn.relu(h_fc1)

#fc2
h_fc2 = tf.matmul(a_fc1, W_fc2)+ b_fc2 #50*500
a_fc = tf.nn.relu(h_fc2)

out1 = a_fc@W_out1 +b_out1
out2_stop = tf.stop_gradient(a_fc)@W_out2 +b_out2

out2_prob = tf.nn.softmax(out2_stop)
tau2_prob = tf.stop_gradient(tf.nn.softmax(out2_stop/FLAGS.tau))

target = tf.nn.embedding_lookup(emb, tf.argmax(y_, -1))
rise = tf.train.exponential_decay(0.1, global_step, 1000, 1.1, staircase=True)
soft_target = tf.nn.softmax(target)
mask = tf.stop_gradient(tf.cast(tf.equal(tf.argmax(out2_stop, 1), tf.argmax(y_, 1)), tf.float32))

L_o1_y = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out1)) #L2
L_o1_emb = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(soft_target), logits=out1)) #L1
L_o2_y = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out2_stop))
L_emb_o2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tau2_prob, logits=target)*mask)/(tf.reduce_sum(mask)+1e-8)
L_re = tf.reduce_sum(tf.nn.relu(tf.reduce_sum(out2_prob*y_,-1)-0.9))

alpha = FLAGS.alpha
if FLAGS.mode=='baseline':
    loss = L_o1_y
elif FLAGS.mode =='emb':
    loss = alpha*L_o1_y + (1-alpha)*L_o1_emb +L_o2_y +L_emb_o2 +L_re

train_step = tf.train.AdamOptimizer().minimize(loss, global_step)
correct_prediction = tf.equal(tf.argmax(out1, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
correct_count = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

def main(_):
    batch_size = FLAGS.batch_size
    if not os.path.isdir('mlp_results'):
                    os.mkdir('mlp_results')
    log = open('./mlp_results/'+FLAGS.num+FLAGS.mode+'.txt','a')
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        valid_data = mnist.validation  #5000
        ITERATION_COUNT = 10000
        ACCURACY_CHECK = 500
        train_epoch = 15
        train_iter_num = mnist.train.num_examples//batch_size
        best=0.0
        ibest=0
        tbest=0.0
        for epoch in range(train_epoch):
            print(str(epoch)+' '+str(batch_size))
            start = time()
            for i in range(1,train_iter_num+1):
                #train:
                batch = mnist.train.next_batch(batch_size)
                _ ,r= sess.run([train_step, rise], feed_dict={
                        x: batch[0],
                        y_: batch[1],
                        is_training: True,
                    })
    
                #DEV
                if (i*batch_size) % 11000 == 0:
                    #print('lr ', lr)
                    loss0, valid_acc, emb_save = sess.run([L_o1_y, accuracy, emb],feed_dict={
                        x: valid_data.images,
                        y_: valid_data.labels,
                        is_training: False,
                    })
                    print(valid_acc)
    
                    if valid_acc >= best:
                        best = valid_acc
                        ibest = epoch
                    #Test:
                    test_data = mnist.test #10000
                    test_size = 1000
                    test_iter_num = test_data.num_examples//test_size
                    all_right = 0.0
                    for _ in range(test_iter_num):
                        batch = test_data.next_batch(test_size, shuffle=False)
                        corrects = sess.run(correct_count, feed_dict={
                                    x: batch[0],
                                    y_: batch[1],
                                    is_training: False,
                                })
                        all_right+=corrects
                    test_acc = all_right/test_data.num_examples
                    if valid_acc >= best: tbest = test_acc
                    log.write(str(epoch)+'/'+str(i) +'\t'+str(valid_acc)+'\t'+str(test_acc)+'\t'+str(loss0)+'\n')
                    #else:
                    #    log.write(str(epoch)+'/'+str(i) +'\t'+str(valid_acc)+'\t'+'#'+'\t'+str(loss0)+'\n')
                    log.flush()
            log.write(str(ibest)+'\t'+str(best)+'\t'+str(tbest)+'\t*\n')
            log.write('**************************************************\n')
        pickle.dump(emb_save, open('mlp_embed.pkl', 'wb'))
        log.write('\n\n\n\n')

if __name__ == '__main__':
    tf.app.run()