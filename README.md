# :sparkles: :robot: Workshop: Machine Learning for the Arts :robot: :sparkles:

:construction: Work in Progress - Check Back Later !

:computer: On the web : https://easternbloc.ca/en/lab/workshops/machine-learning-an-introduction-with-python

:calendar: On Facebook : https://www.facebook.com/events/608059859997028/

---

# :sparkles: :robot: Atelier: l'apprentissage machine pour les arts :robot: :sparkles:

:construction: En cours de préparation - revenez plus tard !

:computer: Sur l'Internet : https://easternbloc.ca/fr/laboratoire/ateliers/machine-learning-python

:calendar: Sur Facebook : https://www.facebook.com/events/608059859997028/

---

:email: Contact: automachine.art@gmail.com

---


# WIP Check Back Later

## Introduction

Hello, this is the source repository for the AI applications for Art workshop held Feb 8th at Eastern bloc, Montreal. Content will be French and English.

### PRÉREQUIS
  Avoir accès à un ordinateur portable lors de l’atelier (nous ne fournissons pas d’ordinateur) / Avoir des connaissances pratiques (débutant-intermédiaire) en programmation / Être familier.ère avec Python 3 /Être familier.ère avec la ligne de commande (Terminal) / Si possible, avoir installé Python 3 sur son ordinateur portable / choses qui ne seront pas développées par l’atelier: compétences générale en programmation, connaissances mathématiques, comment installer Python 3.

### DESCRIPTION
  Cet atelier bilingue d’introduction à l’apprentissage machine pour les arts est une exploration des techniques d’intelligence artificielle (IA) conçue pour les débutant.e.s. Un mélange de travaux pratiques et d’exposés théoriques composeront l’atelier, afin de permettre aux participant.e.s de développer une vue d’ensemble du domaine de l’IA. L’objectif est de familiariser les participant.e.s avec les concepts, le vocabulaire et les algorithmes d’intelligence artificielle à partir d’exemples concrets.

### PREREQUISITE
  Be able to bring your own laptop computer to the workshop (no other device will be available) / Have beginner-to-intermediate level skills in programming / Be at least familiar with Python3 / If possible, have Python 3 installed on your laptop / be familiar with running commands on the terminal / things that won’t be covered in the workshop include: general programming skills, mathematical skills, Python 3 setup

### DESCRIPTION
  This machine learning (ML) meets arts bilingual workshop is aimed at building working knowledge of artificial intelligence for newcomers. A mix of theory and practice, the workshop builds a global view of AI for the arts through examples. Our objective: to have participants be at ease with the main concepts, vocabulary and algorithms of artificial intelligence (AI) by working together on practical projects.

## Schedule
| Time  | Duration | Task
| -----:|:----------:|----------------------------------------------------------------------------
| 9:00  | .5 | Set the general context of ML in the arts scene with a historical perspective.
| 9:30  | .5 | Explain the main ML framework to be used. Go through a basic demo.
| 10:00 | 1h | Help people in setting up their environments on their laptops.
| 11:00 | .5 | Present an introductory ML model (e.g., an image classifier using
|       |    | linear classification using the MNIST dataset).
| 11:30 | .5 | Accompany people as they implement this first ML model.
| 12:00 |    | 1h lunch break.
| 13:00 | .5 | Talk about ML possibilities and shortcomings.
| 13:30 | .5 | Introduce a second ML model based on the focus of choice (e.g.,
|       |    |     sequence-to-sequence, text generation or sound generation).
| 14:00 | 2h | Accompany people as they explore an ML application of their choice.
| 16:00 | .5 | Demystify ML (e.g., how much data do you actually need?).
| 16:30 | .5 | Expansion session: inspirational work (preferably by an Eastern Bloc member).
| 17:00 | inf| Networking session (open to people from outside the workshop).

## Repository Overview

We will be making use of the `utils/` directory throughout the day.

```
.
├── advanced_techinques                <- at the end of the day
│   └── nlp_transformer
├── basic_demos                        <- morning content, note these ones are mostly to verify your environment config.
├── introductory_model                 <- before lunch break
├── README.md                          <- this file
└── utils                              <- tools we'll be using throughout the day
    ├── fetch_text.sh
    └── README

```

### Environment Setup

 For this workshop, we will be using the Python programming language as well as TensorFlow and PyTorch, two deep learning libraries developed for Python. To setup this programming environment, we will be using [Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary), a minimal version of Conda, that includes only `conda`, `Python`, the packages they depend on, and a small number of other useful packages. `conda` is a powerful package manager and environment manager that you use with command line commands in a terminal window.

 To install `conda`, [choose the version for your operating system](https://docs.conda.io/en/latest/miniconda.html). Once conda has been installed, verify that it has been correctly installed by running `conda list` in your terminal. A list of packages should be shown.

 The next step is to install Python, TensorFlow, and PyTorch in a conda "virtual environment". Virtual environments are self-contained programming environments that can have different versions of Python and/or packages installed in them. Switching or moving between environments is called activating the environment. After installing conda, it may need to be initialized by running `conda init bash` (if you're not in windows) in your terminal window. By default, conda installs a base environment that has Python installed. In your terminal, run `conda activate` to activate this base environment. Verify that python has been installed by running `python` in your terminal. This will start the python interactive shell:

 ```
 Python 3.7.5 (default, Oct 25 2019, 10:52:18) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

 To exit the interactive shell, type `exit()`.
 
 To install TensorFlow and PyTorch in a custom conda environment, follow these steps in your terminal:

 ```
conda create -n automachine
conda activate automachine
conda install tensorflow
conda install pytorch torchvision -c pytorch
 ```

 Verify that both TensorFlow and PyTorch have been correctly installed. Follow the below steps in the python interactive shell:

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

 To remove the created conda environment, run `conda remove --name automachine --all` in your terminal.
