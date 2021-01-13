import os
import pickle

import numpy as np
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer

import utils

torch.backends.cudnn.benchmark = True


def run_training():

    args = utils.get_args()

    # Remember to seed!
    seed_everything(args.seed)

    # Setup data module for training
    dm, args = utils.get_data(args)

    # Setup model
    model = utils.get_model(args)

    # Setup callbacks
    loggers, callbacks = utils.get_loggers_callbacks(args, model)

    # Define trainer object from arguments
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=loggers, callbacks=callbacks)

    # ================================================================================================================
    # LEARNING RATE FINDER ROUTINE
    # ----------------------------------------------------------------------------------------------------------------
    if args.lr_finder:
        lr_finder = trainer.tuner.lr_find(model, datamodule=dm)
        fig = lr_finder.plot(suggest=True)
        fig.savefig("results/lr_finder/test.png")
        return
    # ================================================================================================================

    # Fit model using trainer
    trainer.fit(model, dm)

    return 0

    # Return results on eval data
    predictions = trainer.test(
        # model,
        test_dataloaders=dm.val_dataloader(),
        # ckpt_path="best",
        # ckpt_path=trainer.checkpoint_callback.best_model_path,
        verbose=False,
    )
    if not model.use_ddp or (model.use_ddp and torch.distributed.get_rank() == 0):
        predictions = predictions[0]
        with open(os.path.join(args.save_dir, f"eval_predictions.pkl"), "wb") as pkl:
            pickle.dump(predictions, pkl)

        # eval_windows = [1, 15, 30]
        # eval_windows = [1, 2]  # STAGES predicts every 15 s
        eval_windows = [1]  # MASSC predicts every 30 s for evaluation
        df, cm_sub, cm_tot = utils.evaluate_performance(predictions, evaluation_windows=eval_windows, cases=["all"])
        best_acc = []
        best_kappa = []
        with np.printoptions(precision=3, suppress=True):
            s = ""
            for eval_window in cm_tot.keys():
                # print()
                s += "\n"
                s += f"Evaluation window - {eval_window} s\n"
                s += "---------------------------------\n"
                for case in cm_tot[eval_window].keys():
                    df_ = df.query(f'Window == "{eval_window} s" and Case == "{case}"')
                    s += f"Case: {case}\n"
                    s += f"{cm_tot[eval_window][case]}\n"
                    NP = cm_tot[eval_window][case].sum(axis=1)
                    PP = cm_tot[eval_window][case].sum(axis=0)
                    N = cm_tot[eval_window][case].sum()
                    precision = np.diag(cm_tot[eval_window][case]) / (PP + 1e-10)
                    recall = np.diag(cm_tot[eval_window][case]) / (NP + 1e-10)
                    f1 = 2 * precision * recall / (precision + recall + 1e-10)
                    acc = np.diag(cm_tot[eval_window][case]).sum() / N

                    pe = N ** (-2) * (NP @ PP)
                    kappa = 1 - (1 - acc) / (1 - pe)

                    c = np.diag(cm_tot[eval_window][case]).sum()
                    mcc = (c * N - NP @ PP) / (np.sqrt(N ** 2 - (PP @ PP)) * np.sqrt(N ** 2 - (NP @ NP)))

                    s += "\n"
                    s += f'Precision:\t{df_["Precision"].mean():.3f} +/- {df_["Precision"].std():.3f} \t|\t{precision}\n'
                    s += f'Recall:\t\t{df_["Recall"].mean():.3f} +/- {df_["Recall"].std():.3f} \t|\t{recall}\n'
                    s += f'F1: \t\t{df_["F1"].mean():.3f} +/- {df_["F1"].std():.3f} \t|\t{f1}\n'
                    s += f'Accuracy:\t{df_["Accuracy"].mean():.3f} +/- {df_["Accuracy"].std():.3f} \t|\t{acc:.3f}\n'
                    s += f'Kappa:\t\t{df_["Kappa"].mean():.3f} +/- {df_["Kappa"].std():.3f} \t|\t{kappa:.3f}\n'
                    s += f'MCC:\t\t{df_["MCC"].mean():.3f} +/- {df_["MCC"].std():.3f} \t|\t{mcc:.3f}\n'
                    s += "\n"

                    best_acc.append(acc)
                    best_kappa.append(kappa)
        print(s)
        with open(os.path.join(args.save_dir, "eval_case_results.txt"), "w") as txt_file:
            print(s, file=txt_file)
        df.to_csv(os.path.join(args.save_dir, f"eval_results.csv"))
        with open(os.path.join(args.save_dir, f"eval_confusionmatrix.pkl"), "wb") as pkl:
            pickle.dump({"confusiomatrix_subject": cm_sub, "confusionmatrix_total": cm_tot}, pkl)

        try:
            trainer.logger.experiment.summary["best_acc"] = best_acc
            trainer.logger.experiment.summary["best_kappa"] = best_kappa
        except AttributeError:
            trainer.logger.experiment[1].summary["best_acc"] = best_acc
            trainer.logger.experiment[1].summary["best_kappa"] = best_kappa

    return 0


if __name__ == "__main__":
    run_training()
