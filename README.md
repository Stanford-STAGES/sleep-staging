# sleep-staging


# sc_train.py is run by adding an option with the following format:
Example:
python sc_train.py --model ac_lh_ls_lstm

The ac specifies the CC model configuration, the lh specifies the complexity - high in this case, the ls specifies the window length - 15 seconds in this case, and lstm specifies that the model has memory.

To train a model, the sc_config.py should be changed to match the destination for training files, and similarly, to test a model (which has the same option as training) the destination for testing files should be changed.
