import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=2))
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

data = np.random.randint(100, size=(100000,2))
labels = data[:,0] + data[:,1]
print(data[0:5])
print(labels[0:5])

model.fit(data, labels, epochs=10, batch_size=100)

test_data = np.random.randint(100, size=(10,2))
predictions = model.predict(test_data)
actual_results = test_data[:,0] + test_data[:,1]
print(test_data)
print(actual_results)
print(predictions)
