"""
Model training for DMT2 risk prediction assessment.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import argparse
from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

from args import Training
from data.PMBB import PMBBDataset
from data.accession import AccessionConverter


def train(args: argparse.Namespace) -> None:
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

    data_config = DataConfig(
        target=PMBBDataset.get_target_col_name(),
        continuous_cols=PMBBDataset.get_num_col_names(
            args.use_clinical, args.use_idp
        ),
        categorical_cols=PMBBDataset.get_cat_col_names(
            args.use_clinical, args.use_idp
        )
    )
    trainer_config = TrainerConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        gpus=args.gpu
    )
    optimizer_config = OptimizerConfig(optimizer=args.optimizer)

    model_config = FTTransformerConfig(
        task="regression",
        learning_rate=args.lr,
        target_range=[(4.0, 11.0)]
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

    tabular_model.fit(train=train, validation=val)
    tabular_model.save_model("./")


if __name__ == "__main__":
    train(Training.build_args())
