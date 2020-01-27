# Ces ligne sont des commandes d'importation: elles disent à l'interprète Python
# d'utiliser ces bibliothèques afin de faire fonctionner le code que nous écrivons.
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Le modèle est séquentiel: les couches du réseau de neurones se suivent l'une après l'autre, 
# en séquence.
model = Sequential()

# Nous allons créer une couche dense: chaque neurone dans la couche est connecté à chaque 
# neurone dans la suivante.
model.add(Dense(1, input_dim=2))

# Maintenant, nous compilons le modèle afin de le créer selon les importations de bibliothèques
# réalisées ci-haut.
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

# Ce code crée une liste de nombres aléatoires, que nous allons additionner ensemble.
data = np.random.randint(100, size=(100000,2))

# Comme nous allons apprendre au réseau à additionner, nous savons exactement
# ce que les réponses doivent être.
labels = data[:,0] + data[:,1]
print(data[0:5])
print(labels[0:5])

# Cette ligne apprend au réseau à générer les résultats que nous voulons.
model.fit(data, labels, epochs=10, batch_size=100)

# Maintenant, nous voulons tester le réseau sur des nombres qu'il n'a pas déjà vu,
# sur des combinaisons nouvelles pour lui.
test_data = np.random.randint(100, size=(10,2))
predictions = model.predict(test_data)
actual_results = test_data[:,0] + test_data[:,1]

# Nous imprimons les résultats pour vérifier si le réseau fonctionne.
print("This is what we test it on: ", test_data)
print("These are our expected results:", actual_results)
print("This is what the model output:", predictions)
