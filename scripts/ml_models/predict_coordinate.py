#python scripts/ml_models/predict_coordinate.py --model-type logistic_regression --config configs/predict_coordinate_logistic_regression.yaml --timestep 4000 --model-id v1.0 --coord-re -12 0 0

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_proc.config import load_config
from src.ml_models.coordinate_prediction import predict_coordinate
from src.ml_models.model_io import (
    load_logistic_regression_model,
    load_multilayer_perceptron_classifier_model,
    load_perceptron_model,
    load_pytorch_convolutional_neural_network_classifier_model,
)

MODEL_LOADERS_BY_TYPE = {
    "logistic_regression": load_logistic_regression_model,
    "perceptron": load_perceptron_model,
    "multilayer_perceptron_classifier": load_multilayer_perceptron_classifier_model,
    "cnn": load_pytorch_convolutional_neural_network_classifier_model,
}


def main(model_type, config_path, timestep, model_id, coord_re, file_source=None):
    """
    Predict one VDF from coordinates.

    Parameters
    ----------
    model_type : {"logistic_regression", "perceptron", "multilayer_perceptron_classifier", "cnn"}
        Which trained model type to load.
    config_path : str
        Path to coordinate prediction config.
    timestep : int
        VLSV timestep to read.
    model_id : str
        Trained model identifier.
    coord_re : array-like of float
        Coordinate in Earth radii, given as ``[x, y, z]``.
    file_source : str, optional
        Name of file template source to use.
    """

    config = load_config(config_path)
    result = predict_coordinate(
        config=config,
        timestep=timestep,
        model_id=model_id,
        coord_re=coord_re,
        load_model=MODEL_LOADERS_BY_TYPE[model_type],
        file_source=file_source,
    )

    print(f"Saved plot: {result['output_plot_path']}")
    print(f"Predicted class: {result['predicted_class_name']}")
    print(f"{result['score_name']}: {result['prediction_score']}")
    print(result["model_classes"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict one VDF from coordinates using a trained classifier."
    )

    parser.add_argument(
        "--model-type",
        required=True,
        choices=sorted(MODEL_LOADERS_BY_TYPE),
        help="Which trained model type to load.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to coordinate prediction config.",
    )
    parser.add_argument(
        "--timestep",
        required=True,
        help="VLSV timestep to read, for example 3500.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Trained model identifier, for example v1.0.",
    )
    parser.add_argument(
        "--coord-re",
        type=float,
        nargs=3,
        required=True,
        metavar=("X_RE", "Y_RE", "Z_RE"),
        help="Coordinate in Earth radii.",
    )
    parser.add_argument(
        "--file-source",
        default=None,
        help="File source from config file_templates, for example 2d or 3d_egi.",
    )

    args = parser.parse_args()

    main(
        model_type=args.model_type,
        config_path=args.config,
        timestep=args.timestep,
        model_id=args.model_id,
        coord_re=args.coord_re,
        file_source=args.file_source,
    )
