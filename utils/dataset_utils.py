import datasets


def get_data(args):

    dm = datasets.available_datamodules[args.dataset_type](**vars(args))
    dm.setup()

    try:
        args.cb_weights = dm.train.dataset.cb_weights
    except AttributeError:
        pass

    return dm, args
