import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

# 1. Repeat exercises #1 but instead of having two inputs generate three inputs.
input_HoYin = np.random.uniform(-0.6, 0.6, size=(10,3))
output_HoYin = input_HoYin.sum(1).reshape(10,1)

np.random.seed(1)
nn = nl.net.newff(nl.tool.minmax(input_HoYin), [6,1])
error_progress = nn.train(input_HoYin, output_HoYin, show=15, goal=0.00001)

# 2. Test/Simulate the following test sample [0.2,0.1,0.2] record the results in result #5
print('\nResult #5:')
print(nn.sim([[0.2,0.1,0.2]]))

# 3. Repeat exercise #4 but instead of having two inputs generate three inputs.
input_HoYin = np.random.uniform(-0.6, 0.6, size=(100,3))
output_HoYin = input_HoYin.sum(1).reshape(100,1)

np.random.seed(1)
nn = nl.net.newff(nl.tool.minmax(input_HoYin), [5,3,1])

nn.trainf = nl.train.train_gd
error_progress = nn.train(input_HoYin, output_HoYin, epochs=1000, show=100, goal=0.00001)

plt.figure() 
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')
# 4. Test/Simulate the same test in point 10 i.e [0.2,0.1,0.2] record the results in result #6
print('\nResult #6:')
print(nn.sim([[0.2,0.1,0.2]]))