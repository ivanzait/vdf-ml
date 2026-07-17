#python scripts/ml_models/train_classifier.py --model-type logistic_regression --config configs/train_logistic_regression.yaml --dataset-id 3408_100 --model-id v1.0

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_proc.config import load_config
from src.ml_models.training import (
    train_logistic_regression,
    train_multilayer_perceptron_classifier,
    train_perceptron,
)

TRAIN_FUNCTIONS_BY_MODEL_TYPE = {
    "logistic_regression": train_logistic_regression,
    "perceptron": train_perceptron,
    "multilayer_perceptron_classifier": train_multilayer_perceptron_classifier,
}


def main(model_type, config_path, dataset_id, model_id):
    """
    Train a VDF classifier.

    Parameters
    ----------
    model_type : {"logistic_regression", "perceptron", "multilayer_perceptron_classifier"}
        Which model type to train.
    config_path : str
        Path to training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    """

    train_function = TRAIN_FUNCTIONS_BY_MODEL_TYPE[model_type]
    config = load_config(config_path)
    train_function(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a logistic regression, perceptron, or MLP classifier on VDF data."
    )

    parser.add_argument(
        "--model-type",
        required=True,
        choices=sorted(TRAIN_FUNCTIONS_BY_MODEL_TYPE),
        help="Which model type to train.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to training config.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset identifier.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier.",
    )

    args = parser.parse_args()

    main(
        model_type=args.model_type,
        config_path=args.config,
        dataset_id=args.dataset_id,
        model_id=args.model_id,
    )
