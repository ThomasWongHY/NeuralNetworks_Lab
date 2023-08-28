import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

# 1. Repeat step #1 in exercise #3 (i.e. generate 100 samples)
input_HoYin = np.random.uniform(-0.6, 0.6, size=(100,2))
output_HoYin = input_HoYin.sum(1).reshape(100,1)

# 2. Create a two layer feed forward network i.e two hidden layers the first with 5 neurons and the second with 3 neurons.
# a. epochs=1000
# b. show=100
# c. goal=0.00001
np.random.seed(1)
nn = nl.net.newff(nl.tool.minmax(input_HoYin), [5,3,1])

# 3. Set the training algorithm to Gradient descent backpropogation
nn.trainf = nl.train.train_gd

# 4. Train the network using the 100 data points
error_progress = nn.train(input_HoYin, output_HoYin, epochs=1000, show=100, goal=0.00001)

# 5. Plot the error training size graph
plt.figure() 
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')

# 6. Test / simulate the network by passing the following test values 0.1 and 0.2. Record the result under result #4.
print('\nResult #4:')
print(nn.sim([[0.1, 0.2]]))