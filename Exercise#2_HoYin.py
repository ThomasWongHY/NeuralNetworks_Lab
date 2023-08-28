import numpy as np
import neurolab as nl

# 1. Repeat steps 1-8 except for step #5 create a two layer feed forward network i.e two hidden layers the first with 5 neurons and the second with 3 neurons.
# a. epochs=1000
# b. show=100
# c. goal=0.00001
input_HoYin = np.random.uniform(-0.6, 0.6, size=(10,2))
output_HoYin = input_HoYin.sum(1).reshape(10,1)

np.random.seed(1)

nn = nl.net.newff(nl.tool.minmax(input_HoYin), [5,3,1])

# Set the training algorithm to Gradient descent backpropogation
nn.trainf = nl.train.train_gd

error_progress = nn.train(input_HoYin, output_HoYin, epochs=1000, show=100, goal=0.00001)

# 2. Record the result under result #2
print('\nResult #2:')
print(nn.sim([[0.1, 0.2]]))