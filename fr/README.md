# :sparkles: :robot: Atelier: l'apprentissage machine pour les arts :robot: :sparkles:

Bienvenue à notre pratique ! :smile:

### PRÉREQUIS
  Avoir accès à un ordinateur portable lors de l’atelier (nous ne fournissons pas d’ordinateur) / Avoir des connaissances pratiques (débutant-intermédiaire) en programmation / Être familier.ère avec Python 3 /Être familier.ère avec la ligne de commande (Terminal) / Si possible, avoir installé Python 3 sur son ordinateur portable / choses qui ne seront pas développées par l’atelier: compétences générale en programmation, connaissances mathématiques, comment installer Python 3.

### DESCRIPTION
  Cet atelier bilingue d’introduction à l’apprentissage machine pour les arts est une exploration des techniques d’intelligence artificielle (IA) conçue pour les débutant.e.s. Un mélange de travaux pratiques et d’exposés théoriques composeront l’atelier, afin de permettre aux participant.e.s de développer une vue d’ensemble du domaine de l’IA. L’objectif est de familiariser les participant.e.s avec les concepts, le vocabulaire et les algorithmes d’intelligence artificielle à partir d’exemples concrets.


### Mise-en-place de l'environnement

Pour cet atelier, nous utiliserons le language de programmation Python ainsi que les librairies [TensorFlow](https://www.tensorflow.org) et [PyTorch](https://pytorch.org), qui nous permettrons de faire de l'apprentissage profond.

Pour faire la mise en place de cet environnement de programmation, nous utiliserons [Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary), une version minimaliste de [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html), qui inclut seulement `conda`, `Python`, les packages desquels ils dépendent, et un petit nombre d'autres packages pertinents; `conda`est un gestionnaire de packages et un gestionnaire d'environnement puissant que l'on utilise avec la ligne de commande dans une fenêtre de terminal.

Pour installer `miniconda`, suivez les étapes décrites [ici](https://docs.conda.io/en/latest/miniconda.html). Lorsque `miniconda` est installé, vérifiez l'installation en entrant dans la ligne de commande l'instruction `conda list`: une liste de packages devraient alors apparaître.

Ensuite, nous allons installer Python, TensorFlow et PyTorch dans un environnement virtuel que nous allons créé avec `miniconda`. [1]

Après l'installation de `miniconda`, il est possible que ce soit nécessaire de l'initialiser en roulant `conda init bash` dans la fenêtre de terminal. Par défaut, `miniconda`installe un environnement où Python est installé. Dans votre terminal, exécutez l'instruction `conda activate` pour rendre actif cet environnement de base. Vérifiez que Python a été installé en exécutant l'instruction `python` dans la fenêtre de terminal. Ceci lancera Python et le texte suivant s'affichera dans la fenêtre de terminal:

 ```
 Python 3.7.5 (default, Oct 25 2019, 10:52:18) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

Pour sortir de l'interaction avec Python, exécutez la commande `exit()`.

Pour installer TensorFlow et PyTorch dans un environnement spécifique, suivez les étapes suivantes dans votre terminal:

 ```
 $ conda create -n automachine
 $ conda activate automachine
 $ conda install tensorflow
 $ conda install pytorch torchvision -c pytorch
 ```

Vérifiez que TensorFlow et PyTorch ont bel et bien été installés en suivant les étapes suivantes. Commencez par lancer Python à partir de la fenêtre de terminal. Ensuite, nous allons importer PyTorch avec la ligne de code `import torch`, et de même pour TensorFlow avec `import tensorflow`. 

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
>>> x = tf.constant([[1, 2, 3], [4, 5, 6]])
>>> x
<tf.Tensor: id=0, shape=(2, 3), dtype=int32, numpy=array([[1, 2, 3],[4, 5, 6]], dtype=int32)>
 ```

À la fin de l'atelier, si vous désirez ne plus avoir d'environnement virtuel sur votre ordinateur, il suffira d'effacer cet environnement avec la commande de terminal  `conda remove --name automachine --all`.

Pour désinstaller miniconda: 
- sur StackOverflow: https://stackoverflow.com/questions/29596350/how-to-uninstall-mini-conda-python
- dans la documentation officielle: http://docs.continuum.io/anaconda/install.html#id6

 [1] Les environnements virtuels sont des environnements de programmation *self-contained* qui nous permettent d'utiliser différentes versions de Python et/ou des packages installés. Pour changer d'environnement virtuel, il suffit de l'activer ou de le désactiver.