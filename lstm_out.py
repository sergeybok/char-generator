import numpy as np
import tensorflow as tf

import codecs
import sys

from lstm_cell import *


#n_epochs = 500
#batch_size = 1
seq_len = 60

cell_size = 100

learning_rate = 4e-4

batch_size = 20

num_layers = 3

dictionary = []



def load_data(seqlen, dataset='data/input.txt'):
	raw_text = ""
	with codecs.open(dataset, 'r', encoding='UTF-8') as f:
		raw_text = f.read()

	#raw_text = unicodedata.normalize(raw_text, unicode_str)
	data = list(raw_text)
	data_i = []
	for c in data:
		if (not c in dictionary):
				dictionary.append(c)

	d = np.zeros((len(data)//seqlen, seqlen, len(dictionary)),dtype=np.float32)
	y = np.zeros((len(data)//seqlen, seqlen, len(dictionary)),dtype=np.float32)

	mx = len(dictionary)
	rem = len(d) % seqlen
	for i in range(len(data)//seqlen):
		for cur in range(0,seqlen):
			d[i][cur][dictionary.index(data[i*seqlen + cur])] = 1.0
			y[i][cur][dictionary.index(data[i*seqlen + cur + 1])] = 1.0





	return d, y, len(dictionary)



print('loading data..')



dataX, dataY, mx = load_data(seq_len)

validX = dataX[-10:]
validY = dataY[-10:]

dataX = dataX[:-10]
dataY = dataY[:-10]


cell_size = mx*2
print('mx = %i, cell size = %i' % (mx,cell_size))
print('building model..')


batchX = tf.placeholder(tf.float32,[None, None, mx],name='batchX') # [batchsize, seqlen, feature vector]
batchY = tf.placeholder(tf.float32,[None, None, mx],name='batchY')

sampleX = tf.placeholder(tf.float32,[1,None,mx])



tf_zero_state = tf.zeros([num_layers*2, tf.shape(batchX)[0], cell_size],name='tf_zerostate')
tf_initial_state = tf.placeholder(tf.float32,shape=[num_layers*2,1,cell_size],name='tf_initial_state')

lstm = Stacked_LSTM_Cell_Peep(mx,cell_size,num_layers=num_layers)


lstm_outputs = lstm.add_input(input=batchX,init=tf_zero_state)

tf_state = lstm_outputs
outputs = lstm_outputs[-1]


lstm_outputs = lstm.add_input(input=sampleX,init=tf_initial_state)

s_state = lstm_outputs
s_outputs = lstm_outputs[-1]

W_out = tf.Variable(np.random.normal(0,np.sqrt(2/mx),(cell_size,mx)),dtype=tf.float32,name='last_weight')
b_out = tf.Variable(np.zeros(mx,),dtype=tf.float32)


outputs = tf.scan(fn=lambda p,x:tf.matmul(x,W_out)+b_out,elems=outputs,initializer=tf.zeros((tf.shape(batchX)[0],mx)))
s_outputs = tf.scan(fn=lambda p,x:tf.matmul(x,W_out)+b_out,elems=s_outputs,initializer=tf.zeros((1,mx)))

outputs = tf.transpose(outputs,[1,0,2])
s_outputs = tf.transpose(s_outputs,[1,0,2])


sample_outputs = tf.nn.softmax(s_outputs)

ce_cost = tf.nn.softmax_cross_entropy_with_logits(labels=batchY,logits=outputs)


global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learning_rate, global_step,
                                           10000, 0.6, staircase=True)

optimizer = tf.train.AdamOptimizer(lr)
grads_and_vars = optimizer.compute_gradients(ce_cost)
for g in grads_and_vars:
	tf.clip_by_value(g,-2,2)
#tf.clip_by_value(grads_and_vars,-5,5)
#train_step = optimizer.minimize(ce_cost,global_step=global_step)
train_step = optimizer.apply_gradients(grads_and_vars,global_step=global_step)


saver = tf.train.Saver()




def train(n_epochs=120):
	print('beginning training for {0} epochs..'.format(n_epochs))
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())


	n_batches = dataX.shape[0] // batch_size

	for epoch in range(n_epochs):
		loss = []
		avg_loss = 0
		for b in range(n_batches):
			x_in = dataX[b*batch_size:(b+1)*batch_size,:,:]
			y_in = dataY[b*batch_size:(b+1)*batch_size,:,:]
			_, l = sess.run([train_step,ce_cost],feed_dict={
											batchX:x_in,
											batchY:y_in})
			avg_loss += l.mean()/float(n_batches)

		valid_loss = sess.run([ce_cost],feed_dict={batchX:validX,batchY:validY})
		print('\nepoch = %i || loss = %f || valid loss = %f ============================'%(epoch,avg_loss,valid_loss[0].mean()))
		if True:
			# sample
			start_symbol = np.ones([1,1,mx])
			start_symbol *= 1.0/float(mx)
			sample, STATE = sess.run([sample_outputs, s_state], feed_dict={sampleX: start_symbol,
									tf_initial_state:np.zeros((num_layers*2,1,cell_size))})
			sample_char = sample[0][0]
			i = 1
			while(i < 100):
				sample_in = np.random.choice(np.arange(0,sample_char.shape[0]), p=sample_char)
				sys.stdout.write(dictionary[sample_in])
				sample_char *= 0
				sample_char[sample_in] = 1.0
				#in_state = [STAT[i] for i in range(STATE.shape[0]-1)]
				#STATE[-1] = sample[0]
				sample, STATE = sess.run([sample_outputs, s_state], feed_dict={sampleX: [[sample_char]],
								tf_initial_state:STATE.reshape((num_layers*2,1,cell_size))})
				sample_char = sample[0][0]
				i += 1
			#s += '\n\n'
			sample_in = np.random.choice(np.arange(0,sample_char.shape[0]),p=sample_char)
		sys.stdout.write(dictionary[sample_in])
		print('')
		if epoch %5 == 0:
			saver.save(sess, 'models/lstm_model_ep{0}.ckpt'.format(epoch))



def sample(num_chars=400,model_file='models/lstm_model_final.ckpt',):
	sess = tf.InteractiveSession()
	try:
		saver.restore(sess,model_file)
	except:
		print('Could not restore from {0} check that it is the correct file'.format(model_file))
	print('--------sampling {0} characters from {1}-----------'.format(num_chars,model_file))
	start_symbol = np.ones([1,1,mx])
	start_symbol *= 1.0/float(mx)
	sample, STATE = sess.run([sample_outputs, s_state], feed_dict={sampleX: start_symbol,
								tf_initial_state:np.zeros((num_layers*2,1,cell_size))})
	sample_char = sample[0][0]
	i = 1
	while(i < num_chars):
		sample_in = np.random.choice(np.arange(0,sample_char.shape[0]), p=sample_char)
		sys.stdout.write(dictionary[sample_in])
		sample_char *= 0
		sample_char[sample_in] = 1.0
		sample, STATE = sess.run([sample_outputs, s_state], feed_dict={sampleX: [[sample_char]],
						tf_initial_state:STATE.reshape((num_layers*2,1,cell_size))})
		sample_char = sample[0][0]
		i += 1
	sample_in = np.random.choice(np.arange(0,sample_char.shape[0]),p=sample_char)
	sys.stdout.write(dictionary[sample_in])
	print('')











