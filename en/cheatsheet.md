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

# Python Concepts

  Comments : everything after a '#' symbol is for the human to read, and not for the machine

## Variables

Creates a new variable called 'x' with a value of 3:
    
    x = 3 

Creates a new variable called 'x' that is a Tensor (3d or more matrix):

    x = Tensor() 


## Functions

Creates a new function that takes two parameters, 'my' and 'variables':
    
    def myfunction(my, variables) :
      x = 3 + 5
      return x # give back the result of the calculation

Takes the object 'x' and calls it's function 'train' with 'data' parameters:
    
    x.train(data) 

  
## Imports
      
      from <PACKAGE> import <FUNCTION OR OBJECT>
      
It is declared at the beginning of the file so
    that the python interpreter is able to find the required code to execute your program.

## Indexing
  
    data[start:end]
    
Select entries from a list, tensor, array, or datastructure from the start up to
    but not including the end.


