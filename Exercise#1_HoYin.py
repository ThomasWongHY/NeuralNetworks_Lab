import numpy as np
import neurolab as nl

# 1. Create the training data(input) :
# 2. Using numpy random generate two sets of 10 numbers drawn from the uniform distribution,
# make sure to set the numbers to fall between -0.6 and +0.6. Save these numbers in a 10 
# by 2 ndarray where each set is considered a feature. Name the ndarray input_firstname. 
input_HoYin = np.random.uniform(-0.6, 0.6, size=(10,2))
print(input_HoYin)

# 3. Create the target data (output):
# i. The target is the sum of the two random values for each instance of the input data. 
# ii. i.e ( y = x1+x2)
# iii. Store the output in a ndarray 10 by 1. Name the ndarray output_firstname.
output_HoYin = input_HoYin.sum(1).reshape(10,1)
print(output_HoYin)

# 4. Set the seed = 1.
np.random.seed(1)

# 5. Using neurolab, create a simple neural network with two inputs, 6 neurons in the single layer and one output.
nn = nl.net.newff(nl.tool.minmax(input_HoYin), [6, 1])

# 6. Train the network using the input, output data you created in points 1&2 above.
# a. show=15
# b. goal=0.00001
# 7. Train the network using the 10 data points.
error_progress = nn.train(input_HoYin, output_HoYin, show=15, goal=0.00001)

# 8. Test / simulate the network by passing the following test values 0.1 and 0.2. Record the result under result #1.
print('\nResult #1:')
print(nn.sim([[0.1, 0.2]]))