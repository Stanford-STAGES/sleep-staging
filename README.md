# Description 

This repository represents the sleep staging classification work done using neural networks at Stanford University, and is intended primarily for research and historical reference.  

Those interested in using the sleep staging classification methods that were developed from this should use the primary, [Stanford-STAGES](https://www.github.com/stanford-stages/stanford-stages) repository.


# Branches

Git provides support for multiple _branches_ of development.  Notable branches for this repository include:

1. __Master__ (default)

   The ___master branch___.

1. __Historical__ 

   The ___historical branch___ contains some of the initial sleep staging classification work down using neural networks at Stanford University, and is intended primarily for research and historical reference.  

 
1. __Dev__

   The ___dev branch___ is the development branch for updating and testing changes to the ___master branch___ and is not always stable.      

   **NOTE**: Only Python 3.6 and later is supported. Previous versions of Python have been shown to yield erroneous results. 

# Instructions

__sc_train.py__ is run by adding an option with the following format:


Example:

`python sc_train.py --model ac_lh_ls_lstm`

The ac specifies the CC model configuration, the lh specifies the complexity - high in this case, the ls specifies the window length - 15 seconds in this case, and lstm specifies that the model has memory.

To train a model, the sc_config.py should be changed to match the destination for training files, and similarly, to test a model (which has the same option as training) the destination for testing files should be changed.
