#python scripts/ml_models/predict_region.py --model-type logistic_regression --config configs/predict_region_logistic_regression.yaml --start-timestep 1300 --n-timesteps 100 --model-id v1.0 --file-source 3d

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_proc.config import load_config
from src.ml_models.coordinate_prediction import predict_region_timesteps
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


def main(
    model_type,
    config_path,
    start_timestep,
    n_timesteps,
    model_id,
    n_jobs=None,
    file_source=None,
):
    """
    Predict VDF cells in a spatial region for multiple timesteps.

    Parameters
    ----------
    model_type : {"logistic_regression", "perceptron", "multilayer_perceptron_classifier", "cnn"}
        Which trained model type to load.
    config_path : str
        Path to region prediction config.
    start_timestep : int
        First VLSV timestep to read.
    n_timesteps : int
        Number of consecutive timesteps to predict.
    model_id : str
        Trained model identifier.
    n_jobs : int, optional
        Override for ``prediction.n_jobs``.
    file_source : str, optional
        Name of file template source to use.
    """

    config = load_config(config_path)
    if n_jobs is not None:
        config.setdefault("prediction", {})["n_jobs"] = int(n_jobs)

    results = predict_region_timesteps(
        config=config,
        start_timestep=start_timestep,
        n_timesteps=n_timesteps,
        model_id=model_id,
        load_model=MODEL_LOADERS_BY_TYPE[model_type],
        file_source=file_source,
    )

    total_cells = sum(result["n_selected_cells"] for result in results)
    print(f"Predicted timesteps: {len(results)}")
    print(f"Selected VDF cells: {total_cells}")
    for result in results:
        print(
            f"{result['timestep']}: "
            f"{result['n_selected_cells']} cells -> {result['output_path']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict VDF cells inside a spatial region using a trained classifier."
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
        help="Path to region prediction config.",
    )
    parser.add_argument(
        "--start-timestep",
        type=int,
        required=True,
        help="First VLSV timestep to read, for example 1300.",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        required=True,
        help="Number of consecutive timesteps to predict.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Trained model identifier, for example v1.0.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Override prediction.n_jobs from the config.",
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
        start_timestep=args.start_timestep,
        n_timesteps=args.n_timesteps,
        model_id=args.model_id,
        n_jobs=args.n_jobs,
        file_source=args.file_source,
    )
