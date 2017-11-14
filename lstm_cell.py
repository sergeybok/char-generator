import tensorflow as tf
import numpy as np



class LSTM_Cell:
	def __init__(self,in_size, state_size, weights={}):

		F_W_np = np.random.normal(0,np.sqrt(8/(in_size)),(in_size,state_size))
		F_b_np = np.zeros((state_size,))
		F_U_np = np.random.normal(0,np.sqrt(8/(state_size)),(state_size,state_size))
		F_V_np = np.random.normal(0,np.sqrt(8/(state_size)),(state_size,state_size))

		I_W_np = np.random.normal(0,np.sqrt(8/(in_size)),(in_size,state_size))
		I_U_np = np.random.normal(0,np.sqrt(8/(state_size)),(state_size,state_size))
		I_V_np = np.random.normal(0,np.sqrt(8/(state_size)),(state_size,state_size))
		I_b_np = np.zeros((state_size,))

		C_W_np = np.random.normal(0,np.sqrt(8/state_size),(in_size,state_size))
		C_U_np = np.random.normal(0,np.sqrt(8/state_size),(state_size,state_size))
		C_b_np = np.zeros((state_size,))

		O_W_np = np.random.normal(0,np.sqrt(8/in_size),(in_size,state_size))
		O_U_np = np.random.normal(0,np.sqrt(8/state_size),(state_size,state_size))
		O_V_np = np.random.normal(0,np.sqrt(8/state_size),(state_size,state_size))
		O_b_np = np.zeros((state_size,))


		W_f = tf.Variable(F_W_np,dtype=tf.float32)
		U_f = tf.Variable(F_U_np,dtype=tf.float32)
		b_f = tf.Variable(F_b_np,dtype=tf.float32)

		W_i = tf.Variable(I_W_np,dtype=tf.float32)
		U_i = tf.Variable(I_U_np,dtype=tf.float32)
		b_i = tf.Variable(I_b_np,dtype=tf.float32)

		W_c = tf.Variable(C_W_np,dtype=tf.float32)
		U_c = tf.Variable(C_U_np,dtype=tf.float32)
		b_c = tf.Variable(C_b_np,dtype=tf.float32)

		W_o = tf.Variable(O_W_np,dtype=tf.float32)
		U_o = tf.Variable(O_U_np,dtype=tf.float32)
		b_o = tf.Variable(O_b_np,dtype=tf.float32)

		weights = {'W_f':W_f,
				'U_f':U_f,
				'V_f':V_f,
				'b_f':b_f,
				'W_i':W_i,
				'U_i':U_i,
				'V_i':V_i,
				'b_i':b_i,
				'W_c':W_c,
				'U_c':U_c,
				'b_c':b_c,
				'W_o':W_o,
				'U_o':U_o,
				'V_o':V_o,
				'b_o':b_o}

		self.weights = weights

		def step(prev, x_t):
			C_tm1, H_tm1 = tf.unstack(prev) # h_t_m1, previous outputs

			W_f = self.weights['W_f']
			U_f = self.weights['U_f']
			V_f = self.weights['V_f']
			b_f = self.weights['b_f']
			W_i = self.weights['W_i']
			U_i = self.weights['U_i']
			V_i = self.weights['V_i']
			b_i = self.weights['b_i']
			W_c = self.weights['W_c']
			U_c = self.weights['U_c']
			b_c = self.weights['b_c']
			W_o = self.weights['W_o']
			U_o = self.weights['U_o']
			V_o = self.weights['V_o']
			b_o = self.weights['b_o']

			F_t = tf.nn.sigmoid(tf.matmul(x_t,W_f)+tf.matmul(C_tm1,U_f) + b_f)
			I_t = tf.nn.sigmoid(tf.matmul(x_t,W_i)+tf.matmul(C_tm1,U_i) + b_i)
			O_t = tf.nn.sigmoid(tf.matmul(x_t,W_o)+tf.matmul(C_tm1,U_o) + b_o)

			C_t = F_t * C_tm1 + I_t * tf.nn.tanh(tf.matmul(x_t,W_c) + b_c)
			H_t = O_t * C_t

			return tf.stack([C_t, H_t])

		self.step = step

	def add_input(self,input, init):

		lstm_outputs = tf.scan(fn=self.step,elems=tf.transpose(input,[1,0,2]),
					initializer=init)

		return tf.transpose(lstm_outputs,[1,0,2,3])





class Stacked_LSTM_Cell_Peep:
	def __init__(self,in_size, state_size, num_layers=2, reshape_out=False, weights={}):

		weights = {'W_f':[],
				'U_f':[],
				'V_f':[],
				'b_f':[],
				'W_i':[],
				'U_i':[],
				'V_i':[],
				'b_i':[],
				'W_c':[],
				'U_c':[],
				'b_c':[],
				'W_o':[],
				'U_o':[],
				'V_o':[],
				'b_o':[]}
		F_W_np = np.random.normal(0,np.sqrt(2/(in_size)),(in_size,state_size))
		F_b_np = np.zeros((state_size,))
		F_U_np = np.random.normal(0,np.sqrt(2/(state_size)),(state_size,state_size))
		F_V_np = np.random.normal(0,np.sqrt(2/(state_size)),(state_size))


		I_W_np = np.random.normal(0,np.sqrt(2/(in_size)),(in_size,state_size))
		I_U_np = np.random.normal(0,np.sqrt(2/(state_size)),(state_size,state_size))
		I_V_np = np.random.normal(0,np.sqrt(2/(state_size)),(state_size))
		I_b_np = np.zeros((state_size,))

		C_W_np = np.random.normal(0,np.sqrt(1/state_size),(in_size,state_size))
		C_b_np = np.zeros((state_size,))

		O_W_np = np.random.normal(0,np.sqrt(2/in_size),(in_size,state_size))
		O_U_np = np.random.normal(0,np.sqrt(2/state_size),(state_size,state_size))
		O_V_np = np.random.normal(0,np.sqrt(2/state_size),(state_size))
		O_b_np = np.zeros((state_size,))


		W_f = tf.Variable(F_W_np,dtype=tf.float32,name='W_f_0')
		U_f = tf.Variable(F_U_np,dtype=tf.float32)
		V_f = tf.Variable(F_V_np,dtype=tf.float32)
		b_f = tf.Variable(F_b_np,dtype=tf.float32)

		W_i = tf.Variable(I_W_np,dtype=tf.float32,name='W_i_0')
		U_i = tf.Variable(I_U_np,dtype=tf.float32)
		V_i = tf.Variable(I_V_np,dtype=tf.float32)
		b_i = tf.Variable(I_b_np,dtype=tf.float32)

		W_c = tf.Variable(C_W_np,dtype=tf.float32,name='W_c_0')
		b_c = tf.Variable(C_b_np,dtype=tf.float32)

		W_o = tf.Variable(O_W_np,dtype=tf.float32,name='W_o_0')
		U_o = tf.Variable(O_U_np,dtype=tf.float32)
		V_o = tf.Variable(O_V_np,dtype=tf.float32)
		b_o = tf.Variable(O_b_np,dtype=tf.float32)

		weights['W_f'].append(W_f)
		weights['U_f'].append(U_f)
		weights['V_f'].append(V_f)
		weights['b_f'].append(b_f)
		weights['W_i'].append(W_i)
		weights['U_i'].append(U_i)
		weights['V_i'].append(V_i)
		weights['b_i'].append(b_i)
		weights['W_c'].append(W_c)
		weights['b_c'].append(b_c)
		weights['W_o'].append(W_o)
		weights['U_o'].append(U_o)
		weights['V_o'].append(V_o)
		weights['b_o'].append(b_o)

		for i in range(1,num_layers):
			F_W_np = np.random.normal(0,np.sqrt(1/(state_size)),(state_size,state_size))
			F_b_np = np.zeros((state_size,))
			F_U_np = np.random.normal(0,np.sqrt(1/(state_size)),(state_size,state_size))
			F_V_np = np.random.normal(0,np.sqrt(1/(state_size)),(state_size))


			I_W_np = np.random.normal(0,np.sqrt(1/(state_size)),(state_size,state_size))
			I_U_np = np.random.normal(0,np.sqrt(1/(state_size)),(state_size,state_size))
			I_V_np = np.random.normal(0,np.sqrt(1/(state_size)),(state_size))
			I_b_np = np.zeros((state_size,))

			C_W_np = np.random.normal(0,np.sqrt(1/state_size),(state_size,state_size))
			C_b_np = np.zeros((state_size,))

			O_W_np = np.random.normal(0,np.sqrt(1/state_size),(state_size,state_size))
			O_U_np = np.random.normal(0,np.sqrt(1/state_size),(state_size,state_size))
			O_V_np = np.random.normal(0,np.sqrt(1/state_size),(state_size))
			O_b_np = np.zeros((state_size,))


			W_f = tf.Variable(F_W_np,dtype=tf.float32,name=('W_f_{0}'.format(i)))
			U_f = tf.Variable(F_U_np,dtype=tf.float32,name=('U_f_{0}'.format(i)))
			V_f = tf.Variable(F_V_np,dtype=tf.float32)
			b_f = tf.Variable(F_b_np,dtype=tf.float32)

			W_i = tf.Variable(I_W_np,dtype=tf.float32,name=('W_i_{0}'.format(i)))
			U_i = tf.Variable(I_U_np,dtype=tf.float32,name=('U_i_{0}'.format(i)))
			V_i = tf.Variable(I_V_np,dtype=tf.float32)
			b_i = tf.Variable(I_b_np,dtype=tf.float32)

			W_c = tf.Variable(C_W_np,dtype=tf.float32,name=('W_c_{0}'.format(i)))
			b_c = tf.Variable(C_b_np,dtype=tf.float32)

			W_o = tf.Variable(O_W_np,dtype=tf.float32,name=('W_o_{0}'.format(i)))
			U_o = tf.Variable(O_U_np,dtype=tf.float32,name=('U_o_{0}'.format(i)))
			V_o = tf.Variable(O_V_np,dtype=tf.float32)
			b_o = tf.Variable(O_b_np,dtype=tf.float32)

			weights['W_f'].append(W_f)
			weights['U_f'].append(U_f)
			weights['V_f'].append(V_f)
			weights['b_f'].append(b_f)
			weights['W_i'].append(W_i)
			weights['U_i'].append(U_i)
			weights['V_i'].append(V_i)
			weights['b_i'].append(b_i)
			weights['W_c'].append(W_c)
			weights['b_c'].append(b_c)
			weights['W_o'].append(W_o)
			weights['U_o'].append(U_o)
			weights['V_o'].append(V_o)
			weights['b_o'].append(b_o)


		self.weights = weights

		def step(prev, x_t):
			outputs = []
			cur_in = x_t

			for i in range(num_layers):
				#print('i=%i'%(i))

				C_tm1 = prev[i*2]
				H_tm1 = prev[i*2+1]

				W_f = self.weights['W_f'][i]
				U_f = self.weights['U_f'][i]
				V_f = self.weights['V_f'][i]
				b_f = self.weights['b_f'][i]
				W_i = self.weights['W_i'][i]
				U_i = self.weights['U_i'][i]
				V_i = self.weights['V_i'][i]
				b_i = self.weights['b_i'][i]
				W_c = self.weights['W_c'][i]
				b_c = self.weights['b_c'][i]
				W_o = self.weights['W_o'][i]
				U_o = self.weights['U_o'][i]
				V_o = self.weights['V_o'][i]
				b_o = self.weights['b_o'][i]

				F_t = tf.nn.sigmoid(tf.matmul(cur_in,W_f)+tf.matmul(H_tm1,U_f)+(C_tm1*V_f) + b_f)
				I_t = tf.nn.sigmoid(tf.matmul(cur_in,W_i)+tf.matmul(H_tm1,U_i)+(C_tm1*V_i) + b_i)
				O_t = tf.nn.sigmoid(tf.matmul(cur_in,W_o)+tf.matmul(H_tm1,U_o)+(C_tm1*V_o) + b_o)

				C_t = F_t * C_tm1 + I_t * tf.nn.tanh(tf.matmul(cur_in,W_c) + b_c)
				H_t = O_t * C_t
				cur_in = H_t

				outputs += [C_t, H_t]

			return tf.stack(outputs)

		self.step = step

	def add_input(self,input, init):

		lstm_outputs = tf.scan(fn=self.step,elems=tf.transpose(input,[1,0,2]),
					initializer=init)

		return tf.transpose(lstm_outputs,[1,0,2,3])





class Stacked_LSTM_Cell:
	def __init__(self,in_size, state_size, num_layers=2, reshape_out=False, weights={}):

		weights = {'W_f':[],
				'U_f':[],
				'V_f':[],
				'b_f':[],
				'W_i':[],
				'U_i':[],
				'V_i':[],
				'b_i':[],
				'W_c':[],
				'U_c':[],
				'b_c':[],
				'W_o':[],
				'U_o':[],
				'V_o':[],
				'b_o':[]}
		F_W_np = np.random.normal(0,np.sqrt(8/(in_size)),(in_size,state_size))
		F_b_np = np.zeros((state_size,))
		F_U_np = np.random.normal(0,np.sqrt(8/(state_size)),(state_size,state_size))
		F_V_np = np.random.normal(0,np.sqrt(8/(state_size)),(state_size,state_size))


		I_W_np = np.random.normal(0,np.sqrt(8/(in_size)),(in_size,state_size))
		I_U_np = np.random.normal(0,np.sqrt(8/(state_size)),(state_size,state_size))
		I_V_np = np.random.normal(0,np.sqrt(8/(state_size)),(state_size,state_size))
		I_b_np = np.zeros((state_size,))

		C_W_np = np.random.normal(0,np.sqrt(8/state_size),(in_size,state_size))
		C_U_np = np.random.normal(0,np.sqrt(8/state_size),(state_size,state_size))
		C_b_np = np.zeros((state_size,))

		O_W_np = np.random.normal(0,np.sqrt(8/in_size),(in_size,state_size))
		O_U_np = np.random.normal(0,np.sqrt(8/state_size),(state_size,state_size))
		O_V_np = np.random.normal(0,np.sqrt(8/state_size),(state_size,state_size))
		O_b_np = np.zeros((state_size,))

		Out_W_np = np.random.normal(0,np.sqrt(8/state_size),(state_size,state_size))
		Out_b_np = np.zeros((state_size))


		W_f = tf.Variable(F_W_np,dtype=tf.float32)
		U_f = tf.Variable(F_U_np,dtype=tf.float32)
		b_f = tf.Variable(F_b_np,dtype=tf.float32)

		W_i = tf.Variable(I_W_np,dtype=tf.float32)
		U_i = tf.Variable(I_U_np,dtype=tf.float32)
		b_i = tf.Variable(I_b_np,dtype=tf.float32)

		W_c = tf.Variable(C_W_np,dtype=tf.float32)
		U_c = tf.Variable(C_U_np,dtype=tf.float32)
		b_c = tf.Variable(C_b_np,dtype=tf.float32)

		W_o = tf.Variable(O_W_np,dtype=tf.float32)
		U_o = tf.Variable(O_U_np,dtype=tf.float32)
		b_o = tf.Variable(O_b_np,dtype=tf.float32)

		weights['W_f'].append(W_f)
		weights['U_f'].append(U_f)
		weights['b_f'].append(b_f)
		weights['W_i'].append(W_i)
		weights['U_i'].append(U_i)
		weights['b_i'].append(b_i)
		weights['W_c'].append(W_c)
		weights['U_c'].append(U_c)
		weights['b_c'].append(b_c)
		weights['W_o'].append(W_o)
		weights['U_o'].append(U_o)
		weights['b_o'].append(b_o)

		for i in range(num_layers):
			F_W_np = np.random.normal(0,np.sqrt(8/(state_size)),(state_size,state_size))
			F_b_np = np.zeros((state_size,))
			F_U_np = np.random.normal(0,np.sqrt(8/(state_size)),(state_size,state_size))


			I_W_np = np.random.normal(0,np.sqrt(8/(state_size)),(state_size,state_size))
			I_U_np = np.random.normal(0,np.sqrt(8/(state_size)),(state_size,state_size))
			I_b_np = np.zeros((state_size,))

			C_W_np = np.random.normal(0,np.sqrt(8/state_size),(in_size,state_size))
			C_U_np = np.random.normal(0,np.sqrt(8/state_size),(state_size,state_size))
			C_b_np = np.zeros((state_size,))

			O_W_np = np.random.normal(0,np.sqrt(8/state_size),(state_size,state_size))
			O_U_np = np.random.normal(0,np.sqrt(8/state_size),(state_size,state_size))
			O_b_np = np.zeros((state_size,))


			W_f = tf.Variable(F_W_np,dtype=tf.float32)
			U_f = tf.Variable(F_U_np,dtype=tf.float32)
			b_f = tf.Variable(F_b_np,dtype=tf.float32)

			W_i = tf.Variable(I_W_np,dtype=tf.float32)
			U_i = tf.Variable(I_U_np,dtype=tf.float32)
			b_i = tf.Variable(I_b_np,dtype=tf.float32)

			W_c = tf.Variable(C_W_np,dtype=tf.float32)
			U_c = tf.Variable(C_U_np,dtype=tf.float32)
			b_c = tf.Variable(C_b_np,dtype=tf.float32)

			W_o = tf.Variable(O_W_np,dtype=tf.float32)
			U_o = tf.Variable(O_U_np,dtype=tf.float32)
			b_o = tf.Variable(O_b_np,dtype=tf.float32)

			weights['W_f'].append(W_f)
			weights['U_f'].append(U_f)
			weights['b_f'].append(b_f)
			weights['W_i'].append(W_i)
			weights['U_i'].append(U_i)
			weights['b_i'].append(b_i)
			weights['W_c'].append(W_c)
			weights['U_c'].append(U_c)
			weights['b_c'].append(b_c)
			weights['W_o'].append(W_o)
			weights['U_o'].append(U_o)
			weights['b_o'].append(b_o)


		self.weights = weights

		def step(prev, x_t):
			#C_tm1, H_tm1 = tf.unstack(prev) # h_t_m1, previous outputs
			#out_C_t, out_H_t = None
			outputs = []
			cur_in = x_t

			for i in range(num_layers):
				C_tm1 = prev[i*2]
				H_tm1 = prev[i*2+1]

				W_f = self.weights['W_f'][i]
				U_f = self.weights['U_f'][i]
				b_f = self.weights['b_f'][i]
				W_i = self.weights['W_i'][i]
				U_i = self.weights['U_i'][i]
				b_i = self.weights['b_i'][i]
				W_c = self.weights['W_c'][i]
				U_c = self.weights['U_c'][i]
				b_c = self.weights['b_c'][i]
				W_o = self.weights['W_o'][i]
				U_o = self.weights['U_o'][i]
				b_o = self.weights['b_o'][i]

				F_t = tf.nn.sigmoid(tf.matmul(cur_in,W_f)+tf.matmul(H_tm1,U_f) + b_f)
				I_t = tf.nn.sigmoid(tf.matmul(cur_in,W_i)+tf.matmul(H_tm1,U_i) + b_i)
				C_t = F_t * C_tm1 + I_t * tf.nn.tanh(tf.matmul(cur_in,W_c) + b_c)
				O_t = tf.nn.sigmoid(tf.matmul(cur_in,W_o)+tf.matmul(H_tm1,U_o) + b_o)

				H_t = O_t * C_t

				cur_in = H_t

				outputs += [C_t, H_t]

			return tf.stack(outputs)

		self.step = step

	def add_input(self,input, init):

		lstm_outputs = tf.scan(fn=self.step,elems=tf.transpose(input,[1,0,2]),
					initializer=init)

		return tf.transpose(lstm_outputs,[1,0,2,3])













