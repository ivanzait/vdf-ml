#python scripts/ml_models/train_cnn.py --config configs/train_pytorch_convolutional_neural_network_classifier.yaml --dataset-id 3408_100 --model-id v1.0

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_proc.config import load_config
from src.ml_models.pytorch_cnn import (
    apply_pytorch_cnn_training_overrides,
    train_pytorch_convolutional_neural_network_classifier,
)


def main(
    config_path,
    dataset_id,
    model_id,
    class_weight=None,
    class_weight_by_class=None,
    sampler=None,
    sampler_weight_by_class=None,
):
    """
    Train a PyTorch CNN VDF classifier.

    Parameters
    ----------
    config_path : str
        Path to training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    class_weight : str, optional
        Optional model class-weight override.
    class_weight_by_class : sequence of str, optional
        Optional manual class weights as ``class=value``.
    sampler : str, optional
        Optional training sampler override.
    sampler_weight_by_class : sequence of str, optional
        Optional manual sampler weights as ``class=value``.
    """

    config = load_config(config_path)
    apply_pytorch_cnn_training_overrides(
        config=config,
        class_weight=class_weight,
        class_weight_by_class=class_weight_by_class,
        sampler=sampler,
        sampler_weight_by_class=sampler_weight_by_class,
    )
    train_pytorch_convolutional_neural_network_classifier(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PyTorch CNN classifier on VDF data."
    )
    parser.add_argument("--config", required=True, help="Path to CNN config.")
    parser.add_argument("--dataset-id", required=True, help="Dataset identifier.")
    parser.add_argument("--model-id", required=True, help="Model identifier.")
    parser.add_argument(
        "--class-weight",
        choices=["none", "sqrt_balanced", "balanced", "manual"],
        default=None,
        help="Optional model class-weight override.",
    )
    parser.add_argument(
        "--class-weight-by-class",
        action="append",
        default=None,
        metavar="CLASS=VALUE",
        help="Optional manual class weight. Can be repeated.",
    )
    parser.add_argument(
        "--sampler",
        choices=["none", "balanced", "sqrt_balanced", "manual"],
        default=None,
        help="Optional training sampler override.",
    )
    parser.add_argument(
        "--sampler-weight-by-class",
        action="append",
        default=None,
        metavar="CLASS=VALUE",
        help="Optional manual sampler weight. Can be repeated.",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        dataset_id=args.dataset_id,
        model_id=args.model_id,
        class_weight=args.class_weight,
        class_weight_by_class=args.class_weight_by_class,
        sampler=args.sampler,
        sampler_weight_by_class=args.sampler_weight_by_class,
    )
