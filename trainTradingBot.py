from qAgent import Agent
from utils import *
import sys
import matplotlib.pyplot as plt


stock_name, window_size, episode_count = "^GSPC",10,1000

agent = Agent(window_size)
data = getStockDataVec(stock_name)
print(data)
l = len(data) - 1
batch_size = 32

plt.ion() ## Note this correction
fig=plt.figure()
xAxis=[]
yAxis=[]


for e in range(episode_count + 1):
	print ("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)
	print(state)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print ("Buy: " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))


		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print ("--------------------------------")
			print ("Total Profit: " + formatPrice(total_profit))
			print ("--------------------------------")
			xAxis.append(e)
			yAxis.append(total_profit)
			plt.plot(xAxis, yAxis)
			plt.show()
			plt.pause(0.0001)

		if len(agent.memory) > batch_size:
			agent.update_weights(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))
