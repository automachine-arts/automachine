# Helpful Commands for the Terminal

  `<TAB>` : pressing tab at anytime will automatically complete something you've started typing
    for example, to change to a new directory called 'automachine' you can type "cd aut<TAB>" 
    and it will be completed for you
  
  `<UP ARROW>` : pressing the up arrow will bring up your previous command.

##  Windows

  `dir` : print the contents of the current directory

  `cd <FOLDER NAME>` : change current directory to the named folder


##  Linux/Mac OSX

  `ls` : print the contents of the current directory

  `cd <FOLDER NAME>` : change current directory to the named folder

  `sh <SCRIPT>.sh` : execute the script


# GIT commands

  `git checkout <URL>` : make a local copy of a directory containing source code on a server

  `git pull` : update the local copy with the changes on the server

  `git diff` : show me the changes I have made locally


# CONDA commands

  `conda create <ENVIRONMENT>` : create a new environment

  `conda install <PACKAGE> -c <HOST>` : install the package so it can be included and used in your 
    python code, sometimes a host (location to install from) is specified, but it is usually optional

  `conda activate <ENVIRONMENT>` : change to the named virtual environment, which will have it's own version
    of python installed (eg, 3.7, 3.8) and it's own set of packages available to use in your python code

# Machine Learning Concept Definitions

  Floating Point Number : A real number with a decimal point, approximated by a computer.
    Note: try adding 0.1 three times in your python interpreter, you may find the result suprising!

  Tensors : A tensor is a multi-dimensional matrix, imagine it like a cube in space, with rows and columns, 
    each with a floating point number

  Train : Learn the numerical parameters (the neuron values) that will give the correct outputs given the data

  Epoch : one step of training, which makes a small difference to the model to be closer to the correct output

  Predict : called on a model that has already been trained, on new data to generate the output

  Test Data : After training, this is data the model has never seen, that we use to verify it works

  Validation Data : data that's available to train on, but we use it for testing instead

  Neural Network Designs

    Input Layer : This is usually 

# Python Concepts

  Comments : everything after a '#' symbol is for the human to read, and not for the machine

## Variables

```  
    # creates a new variable called 'x' with a value of 3.
    x = 3 

    # creates a new variable called 'x' that is a Tensor (3d or more matrix)
    x = Tensor() 
```

## Functions

```    
    # creates a new function that takes two parameters, 'my' and 'variables'
    def myfunction(my, variables) 
      x = 3 + 5
      return x # give back the result of the calculation

    # takes the object 'x' and calls it's function 'train' with 'data'
    # parameters
    x.train(data) 
```
  Imports : `from <PACKAGE> import <FUNCTION OR OBJECT>` # is is declared at the beginning of the file so
    that the python interpreter is able to find the required code to execute your program.

  Indexing : `data[start:end]` # select entries from a list, tensor, array, or datastructure from the start up to
    but not including the end.

