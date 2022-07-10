"""
Main DMT2RiskAssessment Model training and testing driver program.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License.
"""
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import torch
import xgboost as xgb

from args import build_args
from data.image_dataset import ImageDataset
from models.mlp import MLP


def main():
    args = build_args()
    A1C_THRESHMIN = 6.4

    dataset = ImageDataset(
        biomarkers_datapath=args.biomarkers_datapath,
        steatosis_datapath=args.steatosis_datapath,
        visceral_fat_datapath=args.visceral_fat_datapath,
        verbose=args.verbose
    )
    if len(args.data_split) != 3 or sum(args.data_split) != 100:
        raise ValueError(
            f"Invalid data split {args.data_split} for (train, val, test)."
        )

    seed = args.seed
    if seed is None:
        seed = int(time.time())
    if args.final:
        train_val_split = int(len(dataset) * (
            args.data_split[0] + args.data_split[1]
        ) / 100)
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_val_split, len(dataset) - train_val_split],
            generator=torch.Generator().manual_seed(seed)
        )
        val_dataset = None
    else:
        train_split = int(len(dataset) * args.data_split[0] / 100)
        val_split = int(len(dataset) * args.data_split[1] / 100)
        test_split = len(dataset) - train_split - val_split
        split = torch.utils.data.random_split(
            dataset,
            [train_split, val_split, test_split],
            generator=torch.Generator().manual_seed(seed)
        )
        train_dataset, val_dataset, test_dataset = split
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, num_workers=args.num_workers
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset, num_workers=args.num_workers
        )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, num_workers=args.num_workers
    )
    if args.verbose:
        print(f"Training Dataset Size: {len(train_dataloader)}")
        if val_dataloader is not None:
            print(f"Validation Dataset Size: {len(val_dataloader)}")
        else:
            print("No validation set used, treating as final model.")
        print(f"Test Dataset Size: {len(test_dataloader)}")

    if args.model.lower() == "xgboost" and args.verbose:
        print("Using XGBoost for T2DM Classification")
        print("Combining training and validation datasets.")
    elif args.model.lower() == "mlr" and args.verbose:
        print("Using Multivariate Linear Regression for A1C Prediction")
        print("Combining training and validation datasets.")

    if args.model.lower() == "xgboost":
        xgbc = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=args.lr,
            gamma=0,
            colsample_bytree=1,
            max_depth=8
        )
    elif args.model.lower() == "mlr":
        mlr = LinearRegression()

    if args.model.lower() in ["xgboost", "mlr"]:
        # Construct training matrices.
        training_features, training_outputs = None, None
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            x = np.array([[
                item.subq_metric_area_mean,
                item.visceral_metric_area_mean,
                item.liver_mean_hu,
                item.spleen_mean_hu
            ]])
            y = np.array([[item.a1c_gt > A1C_THRESHMIN]])
            if training_features is None:
                training_features = x
            else:
                training_features = np.concatenate(
                    (training_features, x), axis=0
                )
            if training_outputs is None:
                training_outputs = y
            else:
                training_outputs = np.concatenate(
                    (training_outputs, y), axis=0
                )
        for i in range(len(val_dataset)):
            item = val_dataset[i]
            x = np.array([[
                item.subq_metric_area_mean,
                item.visceral_metric_area_mean,
                item.liver_mean_hu,
                item.spleen_mean_hu
            ]])
            y = np.array([[item.a1c_gt > A1C_THRESHMIN]])
            training_features = np.concatenate(
                (training_features, x), axis=0
            )
            training_outputs = np.concatenate(
                (training_outputs, y), axis=0
            )

    if args.model.lower() == "xgboost":
        xgbc.fit(training_features, training_outputs.astype(int))
    elif args.model.lower() == "mlr":
        mlr= mlr.fit(training_features, training_outputs)

    if args.model.lower() in ["xgboost", "mlr"]:
        # Construct testing matrices.
        testing_features, testing_outputs = None, None
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            x = np.array([[
                item.subq_metric_area_mean,
                item.visceral_metric_area_mean,
                item.liver_mean_hu,
                item.spleen_mean_hu
            ]])
            y = np.array([[item.a1c_gt > A1C_THRESHMIN]])
            if testing_features is None:
                testing_features = x
            else:
                testing_features = np.concatenate(
                    (testing_features, x), axis=0
                )
            if testing_outputs is None:
                testing_outputs = y
            else:
                testing_outputs = np.concatenate(
                    (testing_outputs, y), axis=0
                )
        if args.model.lower() == "xgboost":
            testing_predictions = xgbc.predict(testing_features)
        else:
            testing_predictions = mlr.predict(testing_features)
            testing_predictions = testing_predictions > A1C_THRESHMIN
            testing_predictions = testing_predictions.astype(int)
        print("Test Results")
        print("Confusion Matrix:")
        print(confusion_matrix(testing_outputs, testing_predictions))
        print("\nClassification Report:")
        print(classification_report(testing_outputs, testing_predictions))
        print(
            "\nAccuracy Score: ",
            round(accuracy_score(testing_outputs, testing_predictions), 5)
        )
    elif args.model.lower() == "mlp":
        loss_fn = torch.nn.MSELoss()
        model = MLP(
            in_chans=4,
            out_chans=1,
            chans=args.chans,
            num_layers=args.num_layers,
            activation=args.activation
        )
        if args.verbose:
            print("Using MLP Model for A1C Prediction")
            print(
                "Number of Trainable Parameters:",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
                "\n"
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Model Training.
        N = args.log_train_loss_every_n_steps
        for epoch in range(args.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                optimizer.zero_grad()
                features = torch.Tensor([
                    data.subq_metric_area_mean,
                    data.visceral_metric_area_mean,
                    data.liver_mean_hu,
                    data.spleen_mean_hu
                ])
                outputs = model(features)
                loss = loss_fn(outputs, torch.Tensor([data.a1c_gt]))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if N > 0 and i % N == N - 1:
                    print(
                        f"Epoch {epoch}, Step {i + 1} Loss:",
                        f"{running_loss / N:5f}",
                        flush=True
                    )
                    running_loss = 0.0

            # Model Validation.
            if val_dataloader is not None:
                with torch.no_grad():
                    model.eval()
                    val_loss = 0.0
                    for i, data in enumerate(val_dataloader, 0):
                        features = torch.Tensor([
                            data.subq_metric_area_mean,
                            data.visceral_metric_area_mean,
                            data.liver_mean_hu,
                            data.spleen_mean_hu
                        ])
                        outputs = model(features)
                        loss = loss_fn(outputs, torch.Tensor([data.a1c_gt]))
                        val_loss += loss.item()
                    print(
                        f"Epoch {epoch} Validation Loss:",
                        f"{val_loss / len(val_dataloader)}",
                        flush=True
                    )
                model.train()

        # Model Testing.
        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            test_gt, test_predictions = [], []
            for i, data in enumerate(test_dataloader, 0):
                features = torch.Tensor([
                    data.subq_metric_area_mean,
                    data.visceral_metric_area_mean,
                    data.liver_mean_hu,
                    data.spleen_mean_hu
                ])
                output = model(features)
                loss = loss_fn(output, torch.Tensor([data.a1c_gt]))
                test_gt.append(data.a1c_gt > A1C_THRESHMIN)
                test_predictions.append(output > A1C_THRESHMIN)
                test_loss += loss.item()
            test_gt = np.array(test_gt, dtype=int)
            test_predictions = np.array(test_predictions, dtype=int)

        print("Test Results")
        print(
            "Final Test Loss:",
            f"{test_loss / len(test_dataloader)}",
        )
        print("\nConfusion Matrix:")
        print(confusion_matrix(test_gt, test_predictions))
        print("\nClassification Report:")
        print(classification_report(test_gt, test_predictions))
        print(
            "\nAccuracy Score: ",
            round(accuracy_score(test_gt, test_predictions), 5)
        )
    else:
        raise ValueError(f"Unexpected model type {args.model}.")


if __name__ == "__main__":
    main()
