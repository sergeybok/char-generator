import numpy as np
import tensorflow as tf
#import nltk 




n_epochs = 20
batch_size = 1
seq_len = 30

learning_rate = 0.004


dictionary = ['START', 'END']


def load_data(dataset='will-to-power.txt'):
	raw_text = ""
	with open('data/will-to-power.txt', 'r') as f:
		raw_text = f.read()
	raw_text = raw_text.replace('\n',' ')
	raw_text = raw_text.replace('  ', ' ')
	raw_text = raw_text.replace('  ', ' ')
	raw_text = raw_text.replace('  ', ' ')
	raw_text = raw_text.replace('  ', ' ')

	data = list(raw_text)
	data_i = []
	cur_seq = [0]
	m = 0
	for c in data:
		if (not c in dictionary):
				dictionary.append(c)
		if c == '.':
			cur_seq.append(dictionary.index(c))
			cur_seq.append(dictionary.index('END'))
			data_i.append(cur_seq)
			cur_seq = [0]
			continue
		cur_seq.append(dictionary.index(c))

	#d = np.zeros((len(data_i),len(dictionary)),dtype=np.float32)
	mx = len(dictionary)
	for i in range(len(data_i)):
		seq = []
		for j in range(len(data_i[i])):
			feat = np.zeros((mx,),dtype=np.float32)
			feat[data_i[i][j]] = 1.0
			seq.append(feat)
		data_i[i] = np.array(seq)

	return data_i, len(dictionary)


print('loading data..')

d, mx = load_data()


print('building model..')


batchX = tf.placeholder(tf.float32,[None, None, mx],name='batchX') # [batchsize, seqlen, feature vector]
batchY = tf.placeholder(tf.float32,[None, None, mx],name='batchY')

lstm = tf.contrib.rnn.BasicLSTMCell(mx,state_is_tuple=False)


initial_state = tf.zeros([batch_size, lstm.state_size])


outputs, state = tf.nn.dynamic_rnn(cell=lstm,inputs=batchX,initial_state=initial_state)

ce_cost = tf.nn.softmax_cross_entropy_with_logits(labels=batchY,logits=outputs)

optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(ce_cost)


print('beginning training..')
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


#n_batches = d.shape[0] // batch_size
n_batches = len(d) // batch_size

for epoch in range(n_epochs):
	loss = []
	for b in range(n_batches-batch_size-1):
		x_in = [d[i][:-1] for i in range(batch_size*b,batch_size*(b+1))]
		y_in = [d[i][1:] for i in range(batch_size*b,batch_size*(b+1))]
		_, l = sess.run([train_step,ce_cost],feed_dict={
											batchX:x_in,
											batchY:y_in})
		loss.append(l.mean())

	if epoch % 5:
		avg_loss = sum(loss) / len(loss)
		print('epoch = %i || loss = %i '%(epoch,avg_loss))













