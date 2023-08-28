# Neural_Networks_Lab

### Input
Using numpy random generate two sets of 10 numbers drawn from the uniform distribution, and set the numbers to fall between -0.6 and +0.6. Save these numbers in a 10 by 2 ndarray where each set is considered a feature.
```python
input_HoYin = np.random.uniform(-0.6, 0.6, size=(10,2))
```

### Ouput
The target is the sum of the two random values for each instance of the input data, i.e ( y = x1+x2). Store the output in a ndarray 10 by 1.
```python
output_HoYin = input_HoYin.sum(1).reshape(10,1)
```

## Single Layer Feed Forward Network
Using neurolab, create a simple neural network with two inputs, 6 neurons in the single layer and one output.
```python
nn = nl.net.newff(nl.tool.minmax(input_HoYin), [6, 1])
```
Train the network using the input, output data with the parameter (show=15, goal=0.00001)
```python
error_progress = nn.train(input_HoYin, output_HoYin, show=15, goal=0.00001)
```

### Result #1
Below is the prediction of **Single Layer Neural Network** by y = x1 + x2, where x1 = 0.1 and x2 = 0.2.

<img width="200" alt="image" src="https://github.com/ThomasWongHY/Neural_Networks_Lab/assets/86035047/be18424e-40fc-43e6-86bf-0bb3a26a3154">

## Multi-Layer Feed Forward Network
Using neurolab, create a two layer feed forward network i.e two hidden layers the first with 5 neurons and the second with 3 neurons.
```python
nn = nl.net.newff(nl.tool.minmax(input_HoYin), [5,3,1])
```
Set the training algorithm to Gradient descent backpropogation
```python
nn.trainf = nl.train.train_gd
```
Train the network using the input, output data with the parameter (epochs=1000, show=100, goal=0.00001)
```python
error_progress = nn.train(input_HoYin, output_HoYin, epochs=1000, show=100, goal=0.00001)
```
### Result #2
Below is the prediction of **Multi-Layer Feed Forward Network** by y = x1 + x2, where x1 = 0.1 and x2 = 0.2.

<img width="200" alt="image" src="https://github.com/ThomasWongHY/Neural_Networks_Lab/assets/86035047/be18424e-40fc-43e6-86bf-0bb3a26a3154">

## Single Layer Feed Forward Network with 100 training data
### Input
Generate 100 random instances from uniform distribution and fall between -0.6 and +0.6.
```python
input_HoYin = np.random.uniform(-0.6, 0.6, size=(100,2))
```
Using neurolab, create a simple neural network with two inputs, 6 neurons in the single layer and one output.
```python
nn = nl.net.newff(nl.tool.minmax(input_HoYin), [6, 1])
```
Train the network using the input, output data with the parameter (show=15, goal=0.00001)
```python
error_progress = nn.train(input_HoYin, output_HoYin, show=15, goal=0.00001)
```

### Result #3
Below is the prediction of **Single Layer Neural Network**  with 100 instances by y = x1 + x2, where x1 = 0.1 and x2 = 0.2.

<img width="107" alt="image" src="https://github.com/ThomasWongHY/Neural_Networks_Lab/assets/86035047/2c110991-03bb-4425-8030-102020cde8bf">

## Multi-Layer Feed Forward Network with 100 training data
Using neurolab, create a two layer feed forward network i.e two hidden layers the first with 5 neurons and the second with 3 neurons.
```python
nn = nl.net.newff(nl.tool.minmax(input_HoYin), [5,3,1])
```
Set the training algorithm to Gradient descent backpropogation
```python
nn.trainf = nl.train.train_gd
```
Train the network using the input, output data with the parameter (epochs=1000, show=100, goal=0.00001)
```python
error_progress = nn.train(input_HoYin, output_HoYin, epochs=1000, show=100, goal=0.00001)
```
Plot the error training size graph
```python
plt.figure() 
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')
```

![image](https://github.com/ThomasWongHY/Neural_Networks_Lab/assets/86035047/d028ace9-2a7c-4286-ae4e-d3cb46d2093b)

### Result #4
Below is the prediction of **Multi-Layer Feed Forward Network** by y = x1 + x2, where x1 = 0.1 and x2 = 0.2.

<img width="200" alt="image" src="https://github.com/ThomasWongHY/Neural_Networks_Lab/assets/86035047/f453d485-045e-478b-ada2-b2fc87ed72d0">

## Three Input Single layer Feed Forward
Repeat the step of previous exercise but generate three inputs instead of having two inputs.
```python
input_HoYin = np.random.uniform(-0.6, 0.6, size=(10,3))
output_HoYin = input_HoYin.sum(1).reshape(10,1)

np.random.seed(1)
nn = nl.net.newff(nl.tool.minmax(input_HoYin), [6,1])
error_progress = nn.train(input_HoYin, output_HoYin, show=15, goal=0.00001)
```

### Result #5
Below is the prediction of **Three Input Single layer Feed Forward** by y = x1 + x2 + x3, where x1 = 0.2, x2 = 0.1, x3 = 0.2.

<img width="200" alt="image" src="https://github.com/ThomasWongHY/Neural_Networks_Lab/assets/86035047/de0311b3-e387-492a-9bc0-73bca54cd996">

## Three Input Multi-layer Feed Forward with 100 training data
Repeat the step of previous exercise but generate three inputs instead of having two inputs.
```python
input_HoYin = np.random.uniform(-0.6, 0.6, size=(100,3))
output_HoYin = input_HoYin.sum(1).reshape(100,1)

np.random.seed(1)
nn = nl.net.newff(nl.tool.minmax(input_HoYin), [5,3,1])

nn.trainf = nl.train.train_gd
error_progress = nn.train(input_HoYin, output_HoYin, epochs=1000, show=100, goal=0.00001)
```
Plot the error training size graph
```python
plt.figure() 
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')
```

![image](https://github.com/ThomasWongHY/Neural_Networks_Lab/assets/86035047/9e586d9e-499e-4a8d-9978-733c119b0b5a)

### Result #6
Below is the prediction of **Three Input Multi-layer Feed Forward with 100 training data** by y = x1 + x2 + x3, where x1 = 0.2, x2 = 0.1, x3 = 0.2.

<img width="200" alt="image" src="https://github.com/ThomasWongHY/Neural_Networks_Lab/assets/86035047/02bf01bb-79ef-44ab-aca1-5c2b775d24ea">

## Conclusion
Result #1 and #3 are calculated by singer-layer feed forward and Result #2 and #4 are calculated by multi-layer feed forward. Basically, the number of hidden layers should depend on the complexity of the problem, but the problem is not complex in this case (summation of two values). Also, after trying several times of these exercises, I found that the results of singer-layer feed forward is more accurate than those of multi-layer feed forward sometimes.

Result #3 is trained with 100 data and Result #1 and #2 is trained with 10 data. Since more training data allow the neural network model to enhance its coverage for the problem, Result #3 is more closer to 0.3 than Result #1 and #2. However, irrationally large training dataset may cause overfitting problem.

Result #5 and #6 is calculated with 3 inputs by singer-layer feed forward and multi-layer feed forward respectively. Since the complexity of the problem increases with the number of inputs, the result of multi-layer feed forward is more accurate than that of singer-layer feed forward in this case

On balance, I believe that number of hidden layers work better when the complexity of the problem is high. Also, a rational large number of training data benefit the accuracy of the model.
