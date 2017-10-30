from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from time import time
from math import pow
import pickle
import numpy as np
import os

def weight_variable(shape, seed, stddev):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


v = tf.eye(10)
emb = tf.Variable(v, True)
#emb = tf.Variable(pickle.load(open('cnn_embed.pkl','rb')), False)

# Create the model
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("tau", 2, "Softmax temperature")
tf.app.flags.DEFINE_float("alpha", 0.5, "alpha")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch_size")
tf.app.flags.DEFINE_string("data_dir", "./MNIST_data", "data_dir")
tf.app.flags.DEFINE_string("mode", "baseline", "baseline or labelemb?")
tf.app.flags.DEFINE_string("num", "0", "number")
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)
global_step = tf.Variable(0, trainable=False)
x_image = tf.reshape(x, [-1, 28, 28, 1])

#params
W_conv1 = weight_variable([5, 5, 1, 32], seed=32, stddev=0.1)
b_conv1 = bias_variable([32])
W_conv2 = weight_variable([5, 5, 32, 64], seed=344, stddev=0.1) 
b_conv2 = bias_variable([64])
W_fc1 = weight_variable([7 * 7 * 64, 1024], seed=13, stddev=0.1)
b_fc1 = bias_variable([1024])
W_out1 = weight_variable([1024, 10], seed=33, stddev=0.001)
b_out1 = bias_variable([10])
W_out2 = weight_variable([1024, 10], seed=30, stddev=0.001)
b_out2 = bias_variable([10])


#Conv1:
h_conv1 = conv2d(x_image, W_conv1) + b_conv1 #50*28*28*32
relu_conv1 = tf.nn.relu(h_conv1)
h_pool1 = max_pool_2x2(relu_conv1)

#Conv2:
h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2 #50*14*14*64
relu_conv2 = tf.nn.relu(h_conv2)
h_pool2 = max_pool_2x2(relu_conv2)

#Full Connect1:
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1 #50*300
a_fc = tf.nn.relu(h_fc1) #share

out1 = a_fc@W_out1 +b_out1 
out2_stop = tf.stop_gradient(a_fc)@W_out2 +b_out2

out2_prob = tf.nn.softmax(out2_stop)
tau2_prob = tf.stop_gradient(tf.nn.softmax(out2_stop/FLAGS.tau))

target = tf.nn.embedding_lookup(emb, tf.argmax(y_, -1))
#target = tf.nn.softmax(out)@emb
soft_target = tf.nn.softmax(target)
mask = tf.stop_gradient(tf.cast(tf.equal(tf.argmax(out2_stop, 1), tf.argmax(y_, 1)), tf.float32))

L_o1_y = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out1)) #L2
L_o1_emb = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(soft_target), logits=out1)) #L1
L_o2_y = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out2_stop))
L_emb_o2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tau2_prob, logits=target)*mask)/(tf.reduce_sum(mask)+1e-8)
L_re = tf.reduce_sum(tf.nn.relu(tf.reduce_sum(out2_prob*y_,-1)-0.9))
#re = tf.reduce_mean(tf.square(tf.stop_gradient(W_out1)-W_out2)) + tf.reduce_mean(tf.square(tf.stop_gradient(b_out1)-b_out2))

alpha = FLAGS.alpha
if FLAGS.mode=='baseline':
    loss = L_o1_y
elif FLAGS.mode =='emb':
    loss = alpha*L_o1_y + (1-alpha)*L_o1_emb
else:
    loss = alpha*L_o1_y + (1-alpha)*L_o1_emb +L_o2_y +L_emb_o2 +L_re

train_step = tf.train.AdamOptimizer().minimize(loss)

##########################################################################################
correct_prediction1 = tf.equal(tf.argmax(out1, 1), tf.argmax(y_, 1))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
correct_count1 = tf.reduce_sum(tf.cast(correct_prediction1, tf.int32))

correct_prediction2 = tf.equal(tf.argmax(out2_stop, 1), tf.argmax(y_, 1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
correct_count2 = tf.reduce_sum(tf.cast(correct_prediction2, tf.int32))

def main(_):
    batch_size = FLAGS.batch_size
    if not os.path.isdir('cnn_results'):
                os.mkdir('cnn_results')
    log = open('./cnn_results/'+FLAGS.num+FLAGS.mode+'.txt','a')
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        valid_data = mnist.validation  #5000
        ITERATION_COUNT = 10000
        ACCURACY_CHECK = 500
        train_epoch = 30
        train_iter_num = mnist.train.num_examples//batch_size
        best=0.0
        ibest=0
        tbest=0.0
        itbest=0
        rate = 0.5
        for epoch in range(train_epoch):
            print(str(epoch)+' '+str(batch_size))
            start = time()
            print('rate ',rate)
            for i in range(1, train_iter_num+1):
                #rate = 1.0 * pow(0.98, i/2)
                #train:
                batch = mnist.train.next_batch(batch_size)
                _ = sess.run(train_step, feed_dict={
                        x: batch[0],
                        y_: batch[1],
                        is_training: True,
                    })
                
                #DEV
                if (i*batch_size) % 11000 == 0:
                    valid_acc, embb = sess.run([(accuracy1,accuracy2), emb], feed_dict={
                        x: valid_data.images,
                        y_: valid_data.labels,
                        is_training: False,
                    })
                    print(valid_acc)
                    if valid_acc[0] >= best:
                        #best = valid_acc[0]
                        ibest = epoch
                        #Test:
                        test_data = mnist.test #10000
                        test_size = 1000
                        test_iter_num = test_data.num_examples//test_size
                        all_right = 0.0
                        loss1=[]
                        for _ in range(test_iter_num):
                            batch = test_data.next_batch(test_size, shuffle=False)
                            corrects, los1 = sess.run([correct_count1, L_o1_y], feed_dict={
                                        x: batch[0],
                                        y_: batch[1],
                                        is_training: False,
                                    })
                            all_right+=corrects
                            loss1.append(los1)
                        test_acc = all_right/test_data.num_examples
                        #test_acc, loss1 = sess.run([accuracy1, L1_base], feed_dict={x:test_data.images, #y_:test_data.labels, s_training:False})
                        tbest = test_acc
                        loss1 = np.mean(loss1)
                        log.write(str(epoch)+'/'+str(i) +'\t'+str(valid_acc[0])+'\t'+str(test_acc)+'\t'+str(loss1)+'\n')
                    else:
                        log.write(str(epoch)+'/'+str(i) +'\t'+str(valid_acc[0])+'\t'+'#'+'\t'+str(loss1)+'\n')
                    log.flush()
            #print(soft_emb)
            log.write(str(ibest)+'\t'+str(best)+'\t'+str(tbest)+'\t*\n')
            log.write('**************************************************\n')
            #pickle.dump(embb, open('cnn_embed.pkl', 'wb'))
        log.write('\n\n\n\n')
        #print('********************************')

if __name__ == '__main__':
    tf.app.run()