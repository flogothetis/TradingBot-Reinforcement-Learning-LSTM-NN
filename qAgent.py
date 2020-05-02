# @Author : F. LOGOTHETIS
# This script trains the deep Q_learning model. The purpose of the model
# is succefully make [buy, sell , hold] decisions about any stock.
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque



class Agent:
	def __init__(self, time_window, isEvaluation=False, model_name=""):
		self.time_window = time_window
		self.model_name=model_name
		self.isEvaluation=isEvaluation
		#Three action [buy, sell , hold]
		self.actions =3
		self.inventory=[]
		self.memory = deque(maxlen=1000)
		# gamma is QLearning learning rate
		self.gamma= 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		if( isEvaluation):
			self.model = load_model("models/"+ model_name)
		else:
			self.model= self.createModel()

	def createModel(self):
		model = Sequential()
		model.add(Dense(units=64, input_dim=self.time_window, activation = "relu"))
		model.add(Dense(units=32, activation = "relu"))
		model.add(Dense(units=8, activation = "relu"))
		model.add(Dense(self.actions, activation="linear"))
		model.compile(loss="mse", optimizer= Adam (lr=0.001))
		return model

	def act(self,state):
		options = self.model.predict(state)
		return  np.argmax(options[0])

	def update_weights(self, batch_size):
		mini_batch =[]
		memory_capacity= len(self.memory);
		# add to the mini batch the elements from memory
		for i in range (memory_capacity-batch_size+1, memory_capacity):
			mini_batch.append(self.memory[i])
		for state, action , reward, next_state, done in mini_batch:
			target= reward
			if not done:
				target=reward + self.gamma* np.amax(self.model.predict(next_state)[0])

				target_f=self.model.predict(state)
				target_f[0][action]= target
				#Train model
				self.model.fit(state, target_f, epochs=1, verbose=0)






