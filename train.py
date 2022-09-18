"""
Model training for DMT2 risk prediction assessment.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import argparse
import numpy as np
import os
import pandas as pd
import pickle
import pytorch_lightning as pl
from pytorch_tabular import TabularModel
import pytorch_tabular.models as M
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
import random
from sklearn.metrics import accuracy_score
from statsmodels.regression.linear_model import OLS
from typing import Any
import xgboost as xgb

from args import Training
from data.PMBB import PMBBDataset, PMBBDataModule
from data.accession import AccessionConverter
from baseline.pl_module import FCNNModule
from utils import seed_everything


def train(args: argparse.Namespace) -> Any:
    seed_everything(args.seed, use_deterministic=True)

    data_dir = os.path.join("./data", f"seed_{args.seed}")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    train_fn = os.path.join(data_dir, "train_dataset.pkl")
    val_fn = os.path.join(data_dir, "val_dataset.pkl")
    test_fn = os.path.join(data_dir, "test_dataset.pkl")
    existing_data = os.path.isfile(train_fn)
    existing_data = existing_data and os.path.isfile(val_fn)
    existing_data = existing_data and os.path.isfile(test_fn)
    if existing_data:
        with open(train_fn, "rb") as train_f:
            train = pickle.load(train_f)
        with open(val_fn, "rb") as val_f:
            val = pickle.load(val_f)
        with open(test_fn, "rb") as test_f:
            test = pickle.load(test_f)
        if args.verbose:
            print(f"Loaded train dataset partition from {train_fn}.")
            print(f"Loaded validation dataset partition from {val_fn}.")
            print(f"Loaded test dataset partition from {test_fn}.")
    else:
        converter = AccessionConverter(
            pmbb_to_penn_fn=args.pmbb_to_penn_fn,
            penn_to_date_fn=args.penn_to_date_fn
        )
        dataset = PMBBDataset(
            accession_converter=converter,
            filenames=args.datafiles,
            cache_path=args.cache_path,
            verbose=args.verbose,
            identifier=args.identifier,
            seed=args.seed
        )

        train, val, test = dataset.to_tabular_partitions(
            partitions=args.partitions
        )
        train.to_pickle(train_fn)
        val.to_pickle(val_fn)
        test.to_pickle(test_fn)
        if args.verbose:
            print(f"Saved train dataset partition to {train_fn}.")
            print(f"Saved validation dataset partition to {val_fn}.")
            print(f"Saved test dataset partition to {test_fn}.")

    if args.intelligent:
        bmi_scale_factor = 703.0
        for df in [train, val, test]:
            df["BMI"] = bmi_scale_factor * df["WEIGHT_LBS"] / (
                df["HEIGHT_INCHES"] * df["HEIGHT_INCHES"]
            )
            df["HEPATIC_FAT"] = (
                df["LIVER_MEAN_HU"] - df["SPLEEN_MEAN_HU"]
            )
            df.drop(
                labels=[
                    "HEIGHT_INCHES",
                    "WEIGHT_LBS",
                    "LIVER_MEAN_HU",
                    "SPLEEN_MEAN_HU"
                ],
                axis=1,
                inplace=True
            )
    if args.classifier:
        train.RESULT_VALUE_NUM = train.RESULT_VALUE_NUM >= args.A1C_threshmin
        val.RESULT_VALUE_NUM = val.RESULT_VALUE_NUM >= args.A1C_threshmin
        test.RESULT_VALUE_NUM = test.RESULT_VALUE_NUM >= args.A1C_threshmin

    # Weighted random classifier (classification only).
    if args.model == "WRC":
        if not args.classifier:
            err_msg = "Weighted random classifier only implemented for "
            err_msg += "classification task."
            raise NotImplementedError(err_msg)
        a1c_binary = (train.RESULT_VALUE_NUM).astype(int)
        a1c_prop = np.sum(a1c_binary) / a1c_binary.shape[0]
        preds = []
        for _ in test.RESULT_VALUE_NUM:
            preds.append(int(random.random() <= a1c_prop))
        accuracy = accuracy_score((val.RESULT_VALUE_NUM).astype(int), preds)
        print(f"Accuracy: {accuracy:.6f}")
        return a1c_prop
    # Ordinary least squares (regression only).
    elif args.model == "OLS":
        if args.classifier:
            raise NotImplementedError(
                "OLS only implemented for regression task."
            )
        for df in [train, val, test]:
            # Convert categorical variables to one-hot encoding.
            # Gender: Male = 0, Female = 1
            if "Sex" in df.columns.tolist():
                race_dummies = pd.get_dummies(df["Sex"].copy(deep=True))
                df["Sex_num"] = race_dummies["Female"]
                df.drop(labels=["Sex"], axis=1, inplace=True)
            if "RACE_CODE" in df.columns:
                race_dummies = pd.get_dummies(df["RACE_CODE"])
                for cat in race_dummies.columns:
                    if cat.title() in ["Other", "Unknown"]:
                        continue
                    df[cat] = race_dummies[cat]
                df.drop(labels=["RACE_CODE"], axis=1, inplace=True)
            df = df.reindex(sorted(df.columns), axis=1)
        common_cols = set(train.columns.tolist()) & set(val.columns.tolist())
        common_cols = common_cols & set(test.columns.tolist())
        for df in [train, val, test]:
            for col in df.columns:
                if col not in common_cols:
                    df.drop(labels=[col], axis=1, inplace=True)

        ols_model = OLS(
            np.squeeze(train.RESULT_VALUE_NUM.to_numpy()),
            train.drop(
                labels=["RESULT_VALUE_NUM", "RESULT_DATE_SHIFT"], axis=1
            ).to_numpy(),
            missing="raise"
        )
        ols_results = ols_model.fit()
        preds = np.array(ols_model.predict(
            ols_results.params,
            val.drop(
                labels=["RESULT_VALUE_NUM", "RESULT_DATE_SHIFT"], axis=1
            ).to_numpy(),
        ))
        rmse = np.sqrt(np.mean(np.square(val.RESULT_VALUE_NUM - preds)))
        print(f"Validation RMSE Loss: {rmse:.6f}")
        print(f"PCC: {np.corrcoef(val.RESULT_VALUE_NUM, preds)[0, 1]:.6f}")
        return ols_model
    # Gradient-boosted decision tree.
    elif args.model == "XGBoost":
        tree_method = "gpu_hist" if args.gpu is not None else "hist"
        if not args.classifier:
            xgb_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method=tree_method,
                max_depth=8,
                subsample=1,
                reg_alpha=1,
                reg_lambda=4,
                seed=args.seed,
                random_state=args.seed,
                n_estimators=32,
                learning_rate=args.lr,
                gpu_id=args.gpu,
                enable_categorical=True
            )
        else:
            xgb_model = xgb.XGBClassifier(
                objective="binary:logistic",
                tree_method=tree_method,
                max_depth=16,
                subsample=1,
                reg_alpha=2,
                reg_lambda=4,
                seed=args.seed,
                random_state=args.seed,
                learning_rate=args.lr,
                gpu_id=args.gpu,
                n_estimators=32,
                enable_categorical=True,
            )
        # Specify categorical variable types.
        train["RACE_CODE"] = train["RACE_CODE"].astype("category")
        train["Sex"] = train["Sex"].astype("category")
        xgb_model.fit(
            train.drop(
                ["RESULT_VALUE_NUM", "RESULT_DATE_SHIFT"],
                axis=1
            ),
            train.RESULT_VALUE_NUM
        )

        # Specify categorical variable types.
        val["RACE_CODE"] = val["RACE_CODE"].astype("category")
        val["Sex"] = val["Sex"].astype("category")
        preds = xgb_model.predict(
            val.drop(["RESULT_VALUE_NUM", "RESULT_DATE_SHIFT"], axis=1),
            output_margin=args.classifier
        )
        if args.classifier:
            # Apply sigmoid activation.
            preds = 1.0 / (1.0 + np.exp(-1.0 * np.array(preds)))
            num_threshs = 1000.0
            thresholds = np.array(list(range(int(num_threshs)))) / num_threshs
            scores = []
            for thresh in thresholds:
                scores.append(accuracy_score(
                    (val.RESULT_VALUE_NUM).astype(int), preds >= thresh,
                ))
            # Binarize predictions.
            preds = preds >= thresholds[np.argmax(np.array(scores))]
            acc = accuracy_score((val.RESULT_VALUE_NUM).astype(int), preds)
            print(f"Accuracy: {acc:.6f}")
        else:
            rmse = np.sqrt(np.mean(np.square(val.RESULT_VALUE_NUM - preds)))
            print(f"Validation RMSE Loss: {rmse:.6f}")
            print(f"PCC: {np.corrcoef(val.RESULT_VALUE_NUM, preds)[0, 1]:.6f}")
        return xgb_model
    elif args.model == "FCNN":
        data_module = PMBBDataModule(
            data_dir=os.path.join("./data", f"seed_{args.seed}"),
            batch_size=args.batch_size,
            seed=args.seed
        )
        tabular_model = FCNNModule(
            in_chans=data_module.num_features(),
            out_chans=1,
            chans=32,
            num_layers=8,
            activation="ReLU",
            task="classification",
            A1C_threshmin=args.A1C_threshmin,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma
        )
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            auto_select_gpus=True
        )
        trainer.fit(model=tabular_model, datamodule=data_module)
        trainer.test(model=tabular_model, datamodule=data_module)
        return tabular_model

    data_config = DataConfig(
        target=PMBBDataset.get_target_col_name(),
        continuous_cols=PMBBDataset.get_num_col_names(
            args.use_clinical, args.use_idp, args.intelligent
        ),
        categorical_cols=PMBBDataset.get_cat_col_names(
            args.use_clinical, args.use_idp, args.intelligent
        )
    )
    trainer_config = TrainerConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        gpus=args.gpu,
        early_stopping=None
    )
    optimizer_config = OptimizerConfig(
        optimizer=args.optimizer,
        lr_scheduler="StepLR",
        lr_scheduler_params={
            "step_size": args.lr_step_size,
            "gamma": args.lr_gamma
        }
    )

    task = "regression"
    if args.classifier:
        task = "classification"
    if args.model == "FTTransformer":
        model_config = M.FTTransformerConfig(
            task=task,
            learning_rate=args.lr,
            target_range=[args.A1C_range],
            embedding_dropout=args.dropout,
            attn_dropout=args.dropout,
            add_norm_dropout=args.dropout,
            ff_dropout=args.dropout,
            seed=args.seed
        )
    elif args.model == "AutoInt":
        model_config = M.AutoIntConfig(
            task=task,
            learning_rate=args.lr,
            target_range=[args.A1C_range],
            embedding_dropout=args.dropout,
            dropout=args.dropout,
            seed=args.seed
        )
    elif args.model == "NODEModel":
        model_config = M.NodeConfig(
            task=task,
            learning_rate=args.lr,
            target_range=[args.A1C_range],
            embedding_dropout=args.dropout,
            seed=args.seed
        )
    elif args.model == "TabNet":
        model_config = M.TabNetModelConfig(
            task=task,
            learning_rate=args.lr,
            target_range=[args.A1C_range],
            seed=args.seed
        )
    else:
        raise NotImplementedError(
            f"Unrecognized model {args.model} specified."
        )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    tabular_model.fit(train=train, validation=val)
    return tabular_model


if __name__ == "__main__":
    train(Training.build_args())
