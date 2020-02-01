
# Mise-en-place de l'environnement

## Résumé

1. Téléchargez et installez git à partir de git-scm.com/downloads
2. Téléchargez et installez miniconda avec Python 3.7, selon votre système d'exploitation (OS), à partir de docs.conda.io/en/latest/miniconda.html  
3. Exécutez les commandes suivantes dans le Terminal:

        conda create -n automachine
        conda activate automachine
        conda install tensorflow pillow
        conda install pytorch torchvision -c pytorch    
        conda install transformers -c conda-forge


## Informations complètes

Pour cet atelier, nous utiliserons le language de programmation Python ainsi que les librairies TensorFlow et PyTorch, qui nous permettrons de faire de l'apprentissage profond.

Pour faire la mise en place de cet environnement de programmation, nous utiliserons [Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary), une version minimaliste de Conda, qui inclut seulement `conda`, `Python`, les packages desquels ils dépendent, et un petit nombre d'autres packages pertinents; `conda`est un gestionnaire de packages et un gestionnaire d'environnement puissant que l'on utilise avec la ligne de commande dans une fenêtre de terminal.

Pour installer `miniconda`, suivez les étapes décrites [ici, selon votre système d'exploitation (OS)](https://docs.conda.io/en/latest/miniconda.html). Lorsque `miniconda` est installé, vérifiez l'installation en exécutant dans la ligne de commande l'instruction `conda list` : une liste de packages devrait alors apparaître.

Ensuite, nous allons installer Python, TensorFlow et PyTorch dans un environnement virtuel que nous allons créer avec `miniconda`.

Les environnements virtuels sont des environnements de programmation autonomes qui peuvent avoir différentes versions de Python et/ou de paquets installés à l'intérieur. Pour dire que l'on change d'environnement virtuel, on dit qu'on activate l'environnement.

Après l'installation de `miniconda`, il est possible que ce soit nécessaire de l'initialiser en roulant `conda init bash` dans la fenêtre de terminal. Par défaut, `miniconda` installe un environnement où Python est installé. Dans votre terminal, exécutez l'instruction `conda activate` pour rendre actif cet environnement de base. Vérifiez que Python a été installé en exécutant l'instruction `python` dans la fenêtre de terminal. Ceci lancera Python et le texte suivant s'affichera dans la fenêtre de terminal:

 ```
 Python 3.7.5 (default, Oct 25 2019, 10:52:18) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

Pour sortir de l'interaction avec Python, exécutez la commande `exit()`.

Pour installer TensorFlow et PyTorch dans un environnement spécifique, nommé 'automachine', suivez les étapes suivantes dans votre terminal:

 ```
 $ conda create -n automachine
 $ conda activate automachine
 $ conda install tensorflow
 $ conda install pytorch torchvision -c pytorch
 ```

Vérifiez que TensorFlow et PyTorch ont bel et bien été installés en suivant les étapes suivantes. Commencez par lancer Python à partir de la fenêtre de terminal. Ensuite, nous allons importer PyTorch avec la ligne de code `import torch`, et de même pour TensorFlow avec `import tensorflow as tf`. Ensuite, créez des variables en entrant le code suivant dans la fenêtre Terminal:

        x = torch.tensor([[1, 2, 3], [4, 5, 6]])

        y = tf.constant([[1, 2, 3], [4, 5, 6]])   


Enfin, vérifiez que les variables ont été créés en entrant leurs noms et en appuyant sur Enter.

Vous devriez obtenir quelque chose comme ceci:

 ```
 Python 3.7.5 (default, Oct 25 2019, 10:52:18) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> import tensorflow as tf
>>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> x
tensor([[1, 2, 3],
        [4, 5, 6]])
>>> y = tf.constant([[1, 2, 3], [4, 5, 6]])
>>> y
<tf.Tensor: id=0, shape=(2, 3), dtype=int32, numpy=array([[1, 2, 3],[4, 5, 6]], dtype=int32)>
 ```

Maintenant, nous allons sortir de l'interprète Python: entrez `exit()` et appuyez sur Enter.

Finalement, nous installons Transformers en exécutant dans la fenêtre Terminal la commande suivante:

        conda install transformers -c conda-forge

## Désinstaller

Après l'atelier, si vous désirez ne plus avoir d'environnement virtuel sur votre ordinateur, il suffira d'effacer cet environnement avec la commande de terminal  `conda remove --name automachine --all`.

Pour désinstaller miniconda, voici la documentation officielle: http://docs.continuum.io/anaconda/install.html#id6


