# :sparkles: :robot: Workshop: Machine Learning for the Arts :robot: :sparkles:

Welcome to our rehearsal! :smile:

### PREREQUISITE
  Be able to bring your own laptop computer to the workshop (no other device will be available) / Have beginner-to-intermediate level skills in programming / Be at least familiar with Python3 / If possible, have Python 3 installed on your laptop / be familiar with running commands on the terminal / things that won’t be covered in the workshop include: general programming skills, mathematical skills, Python 3 setup

### DESCRIPTION
  This machine learning (ML) meets arts bilingual workshop is aimed at building working knowledge of artificial intelligence for newcomers. A mix of theory and practice, the workshop builds a global view of AI for the arts through examples. Our objective: to have participants be at ease with the main concepts, vocabulary and algorithms of artificial intelligence (AI) by working together on practical projects.


### Environment Setup

 For this workshop, we will be using the Python programming language as well as [TensorFlow](https://www.tensorflow.org) and [PyTorch](https://pytorch.org), two deep learning libraries developed for Python. To setup this programming environment, we will be using [Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary), a minimal version of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html), that includes only `conda`, `Python`, the packages they depend on, and a small number of other useful packages. `conda` is a powerful package manager and environment manager that you use with command line commands in a terminal window.

 To install `conda`, follow the steps located [here](https://docs.conda.io/en/latest/miniconda.html). Once conda has been installed, open a new terminal and verify that it has been correctly installed by running `conda list` in your terminal. A list of packages should be shown.

 The next step is to install Python, TensorFlow, and PyTorch in a conda "virtual environment". Virtual environments are self-contained programming environments that can have different versions of Python and/or packages installed in them. Switching or moving between environments is called activating the environment. After installing conda, it may need to be initialized by running `conda init bash` in your terminal window. By default, conda installs a base environment that has Python installed. This base environment is automatically started upon opening a new terminal window, indicated by the text `(base)`, to the left (or right) of your terminal cursor. If you wish to remove this automatic behaviour, run `conda config --set auto_activate_base false`.
 
If the base environment has not been activated already, in your terminal, run `conda activate` to activate it. Verify that python has been installed by running `python` in your terminal. This will start the python interactive shell:

 ```
 Python 3.7.5 (default, Oct 25 2019, 10:52:18) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

 To exit the interactive shell, type `exit()`.
 
 To install TensorFlow and PyTorch in a custom conda environment, follow these steps in your terminal:

 ```
 $ conda create -n automachine
 $ conda activate automachine
 $ conda install tensorflow
 $ conda install pytorch torchvision -c pytorch
 ```

 Verify that both TensorFlow and PyTorch have been correctly installed. Follow the below steps in the python interactive shell:

 ```
$ python
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
