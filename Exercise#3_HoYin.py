import numpy as np
import neurolab as nl

# 1. Repeat steps 1-3 in exercise #1 but generate this time 100 random instances.
input_HoYin = np.random.uniform(-0.6, 0.6, size=(100,2))
output_HoYin = input_HoYin.sum(1).reshape(100,1)

# 2. Repeat steps 4-8 in exercise #1
np.random.seed(1)

nn = nl.net.newff(nl.tool.minmax(input_HoYin), [6, 1])
error_progress = nn.train(input_HoYin, output_HoYin, show=15, goal=0.00001)

# 3. Record the result as result #3
print('\nResult #3:')
print(nn.sim([[0.1, 0.2]]))