
# These are 'imports', telling the python interpreter to use the code from
# these libraries to make the code we write work
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# The model will be a squential model, meaning each layer is simply run one
# after the other
model = Sequential()

# We will make a dense layer, meaning each neuron in the layer is connected to
# every neuron in the next
model.add(Dense(1, input_dim=2))

# Compiling the model makes it exist using the code imports we chose above.
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

# This creates a list of random numbers that we want to add together
data = np.random.randint(100, size=(100000,2))

# Since we're teaching the network to add numbers, we know exactly what the
# answer should be
labels = data[:,0] + data[:,1]
print(data[0:5])
print(labels[0:5])

# This is the line that trains the network to generate the results we want!
model.fit(data, labels, epochs=10, batch_size=100)

# We want to test it on numbers that we haven't trained it on, combinations it
# hasn't seen before
test_data = np.random.randint(100, size=(10,2))
predictions = model.predict(test_data)
actual_results = test_data[:,0] + test_data[:,1]
print("This is what we test it on: ", test_data)
print("These are our expected results:", actual_results)
print("This is what the model output:", predictions)
