import datasets


def get_data(args):

    dm = datasets.available_datamodules[args.dataset_type](**vars(args))
    dm.setup()

    try:
        args.n_train = len(dm.train.records)
        args.n_eval = len(dm.eval.records)
        args.cb_weights = dm.train.dataset.cb_weights
    except AttributeError:
        pass

    return dm, args
