
### Environment Setup


## TL;DR

Installation - in brief

1. Download and install git from git-scm.com/downloads
2. Download and install miniconda with Python 3.7 for your OS  from docs.conda.io/en/latest/miniconda.html
3. Run these commands in the miniconda terminal:

        conda create -n automachine
        conda activate automachine
        conda install tensorflow pillow
        conda install pytorch torchvision -c pytorch
        conda install transformers -c conda-forge


## Complete version

 For this workshop, we will be using the Python programming language as well as TensorFlow and PyTorch, two deep learning libraries developed for Python. To setup this programming environment, we will be using [Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary), a minimal version of Conda, that includes only `conda`, `Python`, the packages they depend on, and a small number of other useful packages. `conda` is a powerful package manager and environment manager that you use with command line commands in a terminal window.

 To install `conda`, [choose the version for your operating system](https://docs.conda.io/en/latest/miniconda.html). Once conda has been installed, verify that it has been correctly installed by running `conda list` in your terminal. A list of packages should be shown.

 The next step is to install Python, TensorFlow, and PyTorch in a conda "virtual environment". Virtual environments are self-contained programming environments that can have different versions of Python and/or packages installed in them. Switching or moving between environments is called activating the environment. After installing conda, it may need to be initialized by running `conda init bash` (if you're not in windows) in your terminal window. By default, conda installs a base environment that has Python installed. In your terminal, run `conda activate` to activate this base environment. Verify that python has been installed by running `python` in your terminal. This will start the python interactive shell:

 ```
 Python 3.7.5 (default, Oct 25 2019, 10:52:18) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

 To exit the interactive shell, type `exit()` and press Enter.
 
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

Finally, you can install Transformers by running, in the Terminal, the following command:

        conda install transformers -c conda-forge

## Uninstalling

 To remove the created conda environment, run `conda remove --name automachine --all` in your terminal.
