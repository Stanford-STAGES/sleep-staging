# sleep-staging


# sc_train.py is run by adding an option with the following format:
Example:
python sc_train.py --model ac_lh_ls_lstm

The ac specifies the CC model configuration, the lh specifies the complexity - high in this case, the ls specifies the window length - 15 seconds in this case, and lstm specifies that the model has memory.

To train a model, the sc_config.py should be changed to match the destination for training files, and similarly, to test a model (which has the same option as training) the destination for testing files should be changed.


# Tutorial

A tutorial dataset is included with a subset of code and instructions which can be followed to mimic the steps taken in creating the larger production level models used for sleep stage scoring and narcoleps identification.  The tutorial is given in a publically shared [Google Document](https://docs.google.com/document/d/15q7EJgIF3gACFIpNlNDCTn9V9eaijE5ztIAPrYi_fgk)
