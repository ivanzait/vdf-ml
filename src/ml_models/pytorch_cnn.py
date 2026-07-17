import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import os

os.environ["PTNOLATEX"] = "1"

from src.ml_models.training import (
    create_metrics_text,
    create_predictions,
    evaluate_model,
    load_training_data,
    save_training_artifacts,
)


def apply_pytorch_cnn_training_overrides(
    config,
    class_weight=None,
    class_weight_by_class=None,
    sampler=None,
    sampler_weight_by_class=None,
):
    """
    Apply optional command-line class-weight/sampler overrides.

    Parameters
    ----------
    config : dict
        CNN training config to update in place.
    class_weight : str, optional
        Optional model class-weight override.
    class_weight_by_class : sequence of str, optional
        Optional manual class weights as ``class=value``.
    sampler : str, optional
        Optional training sampler override.
    sampler_weight_by_class : sequence of str, optional
        Optional manual sampler weights as ``class=value``.
    """

    if class_weight is not None:
        model_config = config.setdefault("model", {})
        model_config["class_weight"] = class_weight

    if class_weight_by_class:
        model_config = config.setdefault("model", {})
        model_config["class_weights"] = _parse_threshold_overrides(
            class_weight_by_class,
            option_name="class_weight_by_class",
        )

    if sampler is not None:
        sampler_name = str(sampler).lower()
        model_config = config.setdefault("model", {})
        sampler_config = model_config.setdefault("sampler", {})
        if sampler_name in {"none", "false", "off", "disabled"}:
            sampler_config["enabled"] = False
            sampler_config["mode"] = "none"
        elif sampler_name in {"balanced", "sqrt_balanced", "manual"}:
            sampler_config["enabled"] = True
            sampler_config["mode"] = sampler_name
        else:
            raise ValueError(
                "sampler must be one of: none, balanced, "
                "sqrt_balanced, manual"
            )

    if sampler_weight_by_class:
        model_config = config.setdefault("model", {})
        sampler_config = model_config.setdefault("sampler", {})
        sampler_config["class_weights"] = _parse_threshold_overrides(
            sampler_weight_by_class,
            option_name="sampler_weight_by_class",
        )


def _parse_threshold_overrides(values, option_name):
    """
    Parse class-specific threshold overrides.

    Parameters
    ----------
    values : sequence of str
        Threshold values as ``class=value``.
    option_name : str
        Option name used in error messages.

    Returns
    -------
    dict
        Parsed class-specific thresholds.
    """

    if not values:
        return {}

    thresholds = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option_name} values must use class=value")
        class_name, threshold = value.split("=", 1)
        class_name = class_name.strip()
        if not class_name:
            raise ValueError(f"{option_name} class name must not be empty")
        thresholds[class_name] = float(threshold)

    return thresholds


class PyTorchCNNClassifier(nn.Module):
    """
    Classify flattened VDF xz-slice or Hermite-spectra features with a CNN.

    Parameters
    ----------
    input_size : int
        Number of flattened input features.
    channels : sequence of int
        Number of output channels in each convolutional block.
    classifier_size : int
        Number of neurons in the fully connected hidden layer.
    dropout : float
        Dropout probability before the output layer.
    class_labels : array-like of int
        Project labels in model-output order.
    feature_mean : array-like of float
        Training-feature means used for standardization.
    feature_scale : array-like of float
        Training-feature scales used for standardization.
    adaptive_pool_shape : sequence of int, optional
        Spatial output shape of the final adaptive average pooling layer.
        Must have 2 values for ``representation="raw_vdf"`` or 3 values for
        ``representation="hermite"``. Defaults to ``4`` per dimension.
    representation : {"raw_vdf", "hermite"}, optional
        Input feature representation. ``"raw_vdf"`` reshapes the flattened
        feature vector into a square 2D image for ``Conv2d`` layers (the
        square-image constraint requires ``sqrt(input_size)`` to be an
        integer). ``"hermite"`` reshapes it into the 3D ``volume_shape`` cube
        for ``Conv3d`` layers instead.
    volume_shape : sequence of int, optional
        Required when ``representation="hermite"``: the
        ``(order, order, order)`` shape of the Hermite-spectra cube. Its
        product must equal ``input_size``.
    hermite_rotate : bool, optional
        Whether training data was rotated into a ``(B, v_perp, B x v_perp)``
        frame (``src.data_proc.vdf_helpers.get_rotated_vdf``) before the Hermite
        transform. Recorded on the model (and persisted in its checkpoint)
        purely so inference code can replicate the same preprocessing; it
        does not affect the network architecture.
    prediction_batch_size : int, optional
        Number of feature rows predicted at once.
    """

    def __init__(
        self,
        input_size,
        channels,
        classifier_size,
        dropout,
        class_labels,
        feature_mean,
        feature_scale,
        adaptive_pool_shape=None,
        representation="raw_vdf",
        volume_shape=None,
        hermite_rotate=False,
        prediction_batch_size=64,
    ):
        super().__init__()

        self.input_size = int(input_size)
        self.representation = str(representation).strip().lower()
        if self.representation not in {"raw_vdf", "hermite"}:
            raise ValueError("representation must be 'raw_vdf' or 'hermite'")

        if self.representation == "hermite":
            if volume_shape is None:
                raise ValueError(
                    "volume_shape is required when representation is 'hermite'"
                )
            self.volume_shape = tuple(int(value) for value in volume_shape)
            if len(self.volume_shape) != 3 or any(
                value <= 0 for value in self.volume_shape
            ):
                raise ValueError("volume_shape must contain three positive integers")
            if int(np.prod(self.volume_shape)) != self.input_size:
                raise ValueError(
                    "CNN input features must match the product of volume_shape"
                )
            conv_ndim = 3
        else:
            image_size = int(np.sqrt(self.input_size))
            if image_size**2 != self.input_size:
                raise ValueError("CNN input features must form a square image")
            self.volume_shape = (image_size, image_size)
            conv_ndim = 2

        self.hermite_rotate = bool(hermite_rotate)
        self.channels = tuple(int(channel) for channel in channels)
        self.classifier_size = int(classifier_size)
        self.dropout = float(dropout)
        self.classes_ = np.asarray(class_labels, dtype=int)
        self.adaptive_pool_shape = _resolve_adaptive_pool_shape(
            adaptive_pool_shape,
            ndim=conv_ndim,
        )
        self.prediction_batch_size = int(prediction_batch_size)

        if not self.channels or any(channel <= 0 for channel in self.channels):
            raise ValueError("channels must contain positive integers")
        if self.classifier_size <= 0:
            raise ValueError("classifier_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between zero and one")
        if self.prediction_batch_size <= 0:
            raise ValueError("prediction_batch_size must be positive")

        self.register_buffer(
            "feature_mean",
            torch.as_tensor(feature_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_scale",
            torch.as_tensor(feature_scale, dtype=torch.float32),
        )

        if conv_ndim == 3:
            conv_cls, pool_cls, adaptive_pool_cls = (
                nn.Conv3d,
                nn.AvgPool3d,
                nn.AdaptiveAvgPool3d,
            )
        else:
            conv_cls, pool_cls, adaptive_pool_cls = (
                nn.Conv2d,
                nn.AvgPool2d,
                nn.AdaptiveAvgPool2d,
            )

        convolution_layers = []
        input_channels = 1
        for output_channels in self.channels:
            convolution_layers.extend(
                [
                    conv_cls(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    pool_cls(kernel_size=2),
                ]
            )
            input_channels = output_channels

        convolution_layers.append(adaptive_pool_cls(self.adaptive_pool_shape))
        self.convolutions = nn.Sequential(*convolution_layers)
        pooled_size = int(np.prod(self.adaptive_pool_shape))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channels[-1] * pooled_size, self.classifier_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.classifier_size, len(self.classes_)),
        )

    def forward(self, features):
        """
        Return unnormalized class scores for a feature batch.

        Parameters
        ----------
        features : torch.Tensor
            Flattened VDF/Hermite features with shape
            ``(n_samples, input_size)``.

        Returns
        -------
        torch.Tensor
            Unnormalized class scores with one row per sample.
        """

        embeddings = self.forward_embeddings(features)
        embeddings = self.classifier[3](embeddings)
        return self.classifier[4](embeddings)

    def forward_embeddings(self, features):
        """
        Return hidden-layer embeddings for a feature batch.

        Parameters
        ----------
        features : torch.Tensor
            Flattened VDF/Hermite features with shape
            ``(n_samples, input_size)``.

        Returns
        -------
        torch.Tensor
            Hidden-layer embeddings.
        """

        features = (features - self.feature_mean) / self.feature_scale
        images = features.reshape(-1, 1, *self.volume_shape)
        convolution_features = self.convolutions(images)
        flattened_features = self.classifier[0](convolution_features)
        hidden_features = self.classifier[1](flattened_features)
        return self.classifier[2](hidden_features)

    def predict(self, features):
        """
        Predict project class labels.

        Parameters
        ----------
        features : array-like of float
            Flattened VDF/Hermite features.

        Returns
        -------
        numpy.ndarray
            Predicted project class label for each sample.
        """

        class_indices = np.argmax(self.predict_proba(features), axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, features):
        """
        Predict class probabilities in ``classes_`` order.

        Parameters
        ----------
        features : array-like of float
            Flattened VDF/Hermite features.

        Returns
        -------
        numpy.ndarray
            Class probabilities with one row per sample and columns in
            ``classes_`` order.
        """

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(
                "Expected feature matrix with shape "
                f"(n_samples, {self.input_size})"
            )

        device = self.feature_mean.device
        was_training = self.training
        self.eval()
        probabilities = np.empty(
            (len(features), len(self.classes_)),
            dtype=np.float32,
        )
        with torch.inference_mode():
            for start in range(0, len(features), self.prediction_batch_size):
                end = start + self.prediction_batch_size
                feature_batch = torch.as_tensor(
                    features[start:end],
                    dtype=torch.float32,
                    device=device,
                )
                probabilities[start:end] = (
                    torch.softmax(self(feature_batch), dim=1)
                    .cpu()
                    .numpy()
                )

        if was_training:
            self.train()
        return probabilities

    def transform_embeddings(self, features, batch_size=None):
        """
        Extract CNN hidden-layer embeddings.

        Parameters
        ----------
        features : array-like
            Flattened VDF/Hermite features.
        batch_size : int, optional
            Number of feature rows transformed at once.

        Returns
        -------
        numpy.ndarray
            Hidden-layer embeddings.
        """

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(
                "Expected feature matrix with shape "
                f"(n_samples, {self.input_size})"
            )
        if batch_size is None:
            batch_size = self.prediction_batch_size
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        device = self.feature_mean.device
        was_training = self.training
        self.eval()
        embeddings = np.empty(
            (len(features), self.classifier_size),
            dtype=np.float32,
        )
        with torch.inference_mode():
            for start in range(0, len(features), batch_size):
                end = start + batch_size
                feature_batch = torch.as_tensor(
                    features[start:end],
                    dtype=torch.float32,
                    device=device,
                )
                embeddings[start:end] = (
                    self.forward_embeddings(feature_batch).cpu().numpy()
                )

        if was_training:
            self.train()
        return embeddings


def train_pytorch_convolutional_neural_network_classifier(
    config,
    dataset_id,
    model_id,
):
    """
    Train and save a PyTorch convolutional neural network classifier.

    Parameters
    ----------
    config : dict
        Training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    """

    labels_config = config["labels"]
    class_names_by_label = {
        int(label): class_name
        for class_name, label in labels_config.items()
    }
    if len(class_names_by_label) != len(labels_config):
        raise ValueError("Configured class labels must be unique")

    class_labels = np.asarray(sorted(class_names_by_label), dtype=int)
    class_names = [class_names_by_label[label] for label in class_labels]
    if len(class_labels) < 2:
        raise ValueError("At least two configured classes are required")

    model_config = config["model"]
    channels = tuple(
        int(channel)
        for channel in model_config.get("channels", [16, 32, 64])
    )
    configured_adaptive_pool_shape = model_config.get("adaptive_pool_shape")
    classifier_size = int(model_config.get("classifier_size", 64))
    dropout = float(model_config.get("dropout", 0.2))
    class_weight = _resolve_class_weight(model_config.get("class_weight", "none"))
    weight_decay = float(model_config.get("weight_decay", 0.0001))
    learning_rate = float(model_config.get("learning_rate", 0.0003))
    max_epochs = int(model_config.get("max_epochs", 300))
    early_stopping = bool(model_config.get("early_stopping", True))
    patience = int(model_config.get("patience", 15))
    tolerance = float(model_config.get("tolerance", 1e-4))
    random_seed = int(model_config.get("random_state", 1234))

    if weight_decay < 0.0 or learning_rate <= 0.0:
        raise ValueError("weight_decay must be non-negative and learning rate positive")
    if max_epochs <= 0 or patience <= 0 or tolerance < 0.0:
        raise ValueError("Invalid epoch or early-stopping configuration")

    data = load_training_data(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
        target_kind="multiclass",
    )
    adaptive_pool_shape = _resolve_adaptive_pool_shape(
        configured_adaptive_pool_shape,
        ndim=3 if data["representation"] == "hermite" else 2,
    )

    training_state = _train_cnn_model_for_data(
        data=data,
        class_labels=class_labels,
        class_names=class_names,
        channels=channels,
        adaptive_pool_shape=adaptive_pool_shape,
        classifier_size=classifier_size,
        dropout=dropout,
        class_weight=class_weight,
        model_config=model_config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        patience=patience,
        tolerance=tolerance,
        random_seed=random_seed,
        stage_name="final",
    )
    model = training_state["model"]
    training_result = training_state["training_result"]
    class_weights = training_state["class_weights"]
    sampler_info = training_state["sampler_info"]
    device = training_state["device"]
    deterministic = training_state["deterministic"]
    model_batch_size = training_state["model_batch_size"]
    prediction_batch_size = training_state["prediction_batch_size"]

    results = evaluate_model(
        model=model,
        data=data,
        report_labels=class_labels,
        target_names=class_names,
    )
    print("PyTorch convolutional neural network classifier results")
    print(f"Train accuracy: {results['train_accuracy']}")
    print(f"Validation accuracy: {results['validation_accuracy']}")
    print(f"Test accuracy: {results['test_accuracy']}")
    print(results["print_report"])

    predictions = create_predictions(
        metadata=data["metadata"],
        train_indices=data["train_indices"],
        validation_indices=data["validation_indices"],
        test_indices=data["test_indices"],
        y_train=data["y_train"],
        y_validation=data["y_validation"],
        y_test=data["y_test"],
        y_train_pred=results["y_train_pred"],
        y_validation_pred=results["y_validation_pred"],
        y_test_pred=results["y_test_pred"],
    )
    predictions["true_class_name"] = predictions["true_label"].map(
        class_names_by_label
    )
    predictions["predicted_class_name"] = predictions["predicted_label"].map(
        class_names_by_label
    )
    _add_prediction_probabilities(
        predictions=predictions,
        model=model,
        data=data,
        class_labels=class_labels,
        class_names_by_label=class_names_by_label,
    )

    failure_plot_paths = _plot_failure_cases(
        data=data,
        predictions=predictions,
        class_names_by_label=class_names_by_label,
        plot_config=config.get("failure_plots", {}),
    )

    checkpoint_path = (
        data["output_dir"]
        / "pytorch_convolutional_neural_network_classifier.pt"
    )
    save_pytorch_cnn_checkpoint(model, checkpoint_path)

    metric_lines = [
        "Configured classes:",
        *[
            f"  {label}: {class_name}"
            for label, class_name in zip(class_labels, class_names)
        ],
        f"Input volume shape: (1, {', '.join(str(v) for v in model.volume_shape)})",
        f"Convolution channels: {channels}",
        f"Adaptive pool shape: {adaptive_pool_shape}",
        f"Classifier size: {classifier_size}",
        f"Dropout: {dropout}",
        f"Class weight: {class_weight}",
        f"Sampler enabled: {sampler_info['enabled']}",
        f"Sampler mode: {sampler_info['mode']}",
        "Activation: relu",
        "Pooling: average",
        "Optimizer: AdamW",
        f"Weight decay: {weight_decay}",
        f"Model batch size: {model_batch_size}",
        f"Prediction batch size: {prediction_batch_size}",
        f"Learning rate: {learning_rate}",
        f"Max epochs: {max_epochs}",
        f"Early stopping: {early_stopping}",
        "Early stopping metric: validation macro F1",
        f"Patience: {patience}",
        f"Tolerance: {tolerance}",
        f"Random state: {random_seed}",
        f"Device: {device}",
        f"Deterministic algorithms: {deterministic}",
        f"PyTorch version: {torch.__version__}",
        f"Classifier classes: {list(model.classes_)}",
        f"Epochs: {training_result['n_epochs']}",
        f"Best epoch: {training_result['best_epoch']}",
        f"Final training loss: {training_result['final_training_loss']}",
        f"Failure plots saved: {len(failure_plot_paths)}",
    ]
    if failure_plot_paths:
        metric_lines.append(f"Failure plot directory: {data['output_dir'] / 'failure_plots'}")
    if class_weights is not None:
        metric_lines.append(f"Class weights: {class_weights.tolist()}")
    if sampler_info["class_weights"] is not None:
        metric_lines.append(
            f"Sampler class weights: {sampler_info['class_weights'].tolist()}"
        )
    if training_result["best_validation_macro_f1"] is not None:
        metric_lines.append(
            "Best validation macro F1: "
            f"{training_result['best_validation_macro_f1']}"
        )

    save_training_artifacts(
        output_dir=data["output_dir"],
        preprocessing_values={
            "representation": np.asarray(data["representation"]),
            "hermite_rotate": np.asarray(data.get("hermite_rotate", False)),
            "downsample_factor": data["downsample_factor"],
            "dataset_id": dataset_id,
            "model_id": model_id,
            "log_eps": data["log_eps"],
            "batch_size": data["batch_size"],
            "n_jobs": data["n_jobs"],
            "train_fraction": data["train_fraction"],
            "validation_fraction": data["validation_fraction"],
            "gap_timesteps": data["gap_timesteps"],
            "class_labels": class_labels,
            "class_names": np.asarray(class_names),
            "adaptive_pool_shape": np.asarray(adaptive_pool_shape, dtype=int),
        },
        predictions=predictions,
        metrics_text=create_metrics_text(
            title="PyTorch convolutional neural network classifier evaluation",
            dataset_id=dataset_id,
            model_id=model_id,
            data=data,
            results=results,
            extra_lines=metric_lines,
        ),
    )

    print(checkpoint_path)
    print(data["output_dir"] / "metrics.txt")
    print(predictions)


def _add_prediction_probabilities(
    predictions,
    model,
    data,
    class_labels,
    class_names_by_label,
):
    """
    Add class probability columns to the prediction table.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Prediction rows in train, validation, and test order.
    model : PyTorchCNNClassifier
        Trained CNN classifier.
    data : dict
        Training data returned by ``load_training_data``.
    class_labels : numpy.ndarray
        Project labels in model-output order.
    class_names_by_label : dict
        Mapping from integer label to class name.
    """

    probabilities = np.concatenate(
        [
            model.predict_proba(data["X_train_features"]),
            model.predict_proba(data["X_validation_features"]),
            model.predict_proba(data["X_test_features"]),
        ],
        axis=0,
    )
    if len(probabilities) != len(predictions):
        raise ValueError("Prediction probability count does not match predictions")

    label_to_column = {}
    for column_index, label in enumerate(class_labels):
        class_name = class_names_by_label[int(label)]
        column_name = f"probability_{class_name}"
        predictions[column_name] = probabilities[:, column_index]
        label_to_column[int(label)] = column_index

    true_columns = predictions["true_label"].map(label_to_column).to_numpy(dtype=int)
    predicted_columns = predictions["predicted_label"].map(label_to_column).to_numpy(
        dtype=int
    )
    row_indices = np.arange(len(predictions))
    predictions["true_class_probability"] = probabilities[row_indices, true_columns]
    predictions["predicted_class_probability"] = probabilities[
        row_indices,
        predicted_columns,
    ]
    predictions["max_class_probability"] = probabilities.max(axis=1)


def load_pytorch_cnn_checkpoint(
    checkpoint_path,
    device="cpu",
    prediction_batch_size=64,
):
    """
    Load a PyTorch convolutional neural network checkpoint.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Saved checkpoint path.
    device : str or torch.device, optional
        Device used for prediction.
    prediction_batch_size : int, optional
        Number of feature rows predicted at once.

    Returns
    -------
    PyTorchCNNClassifier
        Loaded classifier with a NumPy prediction interface.
    """

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    class_labels = checkpoint["class_labels"]
    if isinstance(class_labels, torch.Tensor):
        class_labels = class_labels.cpu().numpy()
    if len(class_labels) != int(checkpoint["n_classes"]):
        raise ValueError("Checkpoint class-label count does not match model output")

    input_size = int(checkpoint["input_size"])
    model = PyTorchCNNClassifier(
        input_size=input_size,
        channels=checkpoint["channels"],
        classifier_size=int(checkpoint["classifier_size"]),
        dropout=float(checkpoint["dropout"]),
        class_labels=class_labels,
        feature_mean=np.zeros(input_size, dtype=np.float32),
        feature_scale=np.ones(input_size, dtype=np.float32),
        adaptive_pool_shape=checkpoint.get("adaptive_pool_shape"),
        representation=str(checkpoint.get("representation", "raw_vdf")),
        volume_shape=checkpoint.get("volume_shape"),
        hermite_rotate=bool(checkpoint.get("hermite_rotate", False)),
        prediction_batch_size=prediction_batch_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def save_pytorch_cnn_checkpoint(model, checkpoint_path):
    """
    Save model weights, preprocessing values, and architecture information.

    Parameters
    ----------
    model : PyTorchCNNClassifier
        Trained CNN classifier to save.
    checkpoint_path : str or pathlib.Path
        Output path for the PyTorch checkpoint.
    """

    checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "input_size": model.input_size,
        "representation": model.representation,
        "volume_shape": list(model.volume_shape),
        "hermite_rotate": model.hermite_rotate,
        "channels": list(model.channels),
        "classifier_size": model.classifier_size,
        "adaptive_pool_shape": list(model.adaptive_pool_shape),
        "dropout": model.dropout,
        "n_classes": len(model.classes_),
        "class_labels": torch.as_tensor(model.classes_, dtype=torch.int64),
    }

    torch.save(checkpoint, checkpoint_path)


def _train_cnn_model_for_data(
    data,
    class_labels,
    class_names,
    channels,
    adaptive_pool_shape,
    classifier_size,
    dropout,
    class_weight,
    model_config,
    learning_rate,
    weight_decay,
    max_epochs,
    early_stopping,
    patience,
    tolerance,
    random_seed,
    stage_name,
):
    """
    Train a CNN model for the current in-memory training arrays.

    Parameters
    ----------
    data : dict
        Training data returned by ``load_training_data``.
    class_labels : numpy.ndarray
        Project labels in model-output order.
    class_names : list of str
        Class names in model-output order.
    channels : tuple of int
        Number of convolution channels.
    adaptive_pool_shape : tuple of int
        Spatial shape after adaptive pooling.
    classifier_size : int
        Number of hidden-layer neurons.
    dropout : float
        Dropout probability.
    class_weight : str
        Class weighting mode.
    model_config : dict
        CNN model configuration.
    learning_rate : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    max_epochs : int
        Maximum number of training epochs.
    early_stopping : bool
        Whether to use validation macro-F1 early stopping.
    patience : int
        Early-stopping patience.
    tolerance : float
        Minimum validation macro-F1 improvement.
    random_seed : int
        Random seed.
    stage_name : str
        Name printed before training.

    Returns
    -------
    dict
        Trained model and training settings.
    """

    y_train = _encode_labels(data["y_train"], class_labels)
    y_validation = _encode_labels(data["y_validation"], class_labels)
    _encode_labels(data["y_test"], class_labels)

    missing_classes = set(range(len(class_labels))) - set(y_train)
    if missing_classes:
        missing_labels = class_labels[sorted(missing_classes)]
        raise ValueError(
            "Configured classes have no training samples: "
            f"{list(missing_labels)}"
        )

    class_weights = _create_class_weights(
        targets=y_train,
        n_classes=len(class_labels),
        class_weight=class_weight,
        class_names=class_names,
        configured_class_weights=model_config.get("class_weights", {}),
    )
    sampler, sampler_info = _create_training_sampler(
        targets=y_train,
        class_names=class_names,
        sampler_config=model_config.get("sampler", {}),
        random_seed=random_seed,
    )

    scaler = StandardScaler().fit(data["X_train_features"])
    device = _resolve_device(model_config.get("device", "auto"))
    deterministic = bool(model_config.get("deterministic", False))
    _set_random_seed(random_seed, deterministic)

    model_batch_size = _resolve_batch_size(
        model_config.get("batch_size", 32),
        len(y_train),
    )
    prediction_batch_size = _resolve_prediction_batch_size(
        configured_batch_size=model_config.get("prediction_batch_size", 64),
        model_batch_size=model_batch_size,
        n_samples=len(y_train),
    )

    model = PyTorchCNNClassifier(
        input_size=data["X_train_features"].shape[1],
        channels=channels,
        classifier_size=classifier_size,
        dropout=dropout,
        class_labels=class_labels,
        feature_mean=np.asarray(scaler.mean_, dtype=np.float32),
        feature_scale=np.asarray(scaler.scale_, dtype=np.float32),
        adaptive_pool_shape=adaptive_pool_shape,
        representation=data.get("representation", "raw_vdf"),
        volume_shape=data.get("volume_shape"),
        hermite_rotate=data.get("hermite_rotate", False),
        prediction_batch_size=prediction_batch_size,
    ).to(device)

    print("Configured classes:")
    for label, class_name in zip(class_labels, class_names):
        print(f"  {label}: {class_name}")
    print(f"CNN stage: {stage_name}")
    print(f"CNN input volume: {' x '.join(str(v) for v in model.volume_shape)}")
    print(f"Training device: {device}")

    training_result = _fit_model(
        model=model,
        features=data["X_train_features"],
        targets=y_train,
        validation_features=data["X_validation_features"],
        validation_targets=y_validation,
        class_weights=class_weights,
        sampler=sampler,
        device=device,
        batch_size=model_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        tolerance=tolerance,
        patience=patience,
        random_seed=random_seed,
    )
    model.eval()

    return {
        "model": model,
        "training_result": training_result,
        "class_weights": class_weights,
        "sampler_info": sampler_info,
        "device": device,
        "deterministic": deterministic,
        "model_batch_size": model_batch_size,
        "prediction_batch_size": prediction_batch_size,
    }


def _fit_model(
    model,
    features,
    targets,
    validation_features,
    validation_targets,
    class_weights,
    sampler,
    device,
    batch_size,
    learning_rate,
    weight_decay,
    max_epochs,
    early_stopping,
    tolerance,
    patience,
    random_seed,
):
    """
    Fit a CNN classifier.

    Parameters
    ----------
    model : PyTorchCNNClassifier
        CNN classifier to train.
    features : array-like of float
        Flattened training VDF/Hermite features.
    targets : numpy.ndarray
        Encoded training class indices.
    validation_features : array-like of float
        Flattened validation VDF/Hermite features.
    validation_targets : numpy.ndarray
        Encoded validation class indices.
    class_weights : numpy.ndarray or None
        Optional class weights in model-output order.
    sampler : torch.utils.data.Sampler or None
        Optional sampler for selecting training rows.
    device : torch.device
        Device used for model fitting.
    batch_size : int
        Number of training samples in each optimization batch.
    learning_rate : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight-decay coefficient.
    max_epochs : int
        Maximum number of training epochs.
    early_stopping : bool
        Whether to stop using validation macro F1.
    tolerance : float
        Minimum score improvement that resets early-stopping patience.
    patience : int
        Number of epochs without improvement before stopping.
    random_seed : int
        Seed used for shuffled training batches.

    Returns
    -------
    dict
        Epoch count, best epoch and validation score, and final training loss.
    """

    features = np.asarray(features, dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=None if sampler is not None else torch.Generator().manual_seed(
            random_seed
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    if class_weights is None:
        loss_weights = None
    else:
        loss_weights = torch.as_tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    loss_function = nn.CrossEntropyLoss(weight=loss_weights)
    best_score = float("-inf")
    best_state_dict = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = torch.zeros((), dtype=torch.float32, device=device)

        for feature_batch, target_batch in data_loader:
            feature_batch = feature_batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            target_batch = target_batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            optimizer.zero_grad(set_to_none=True)
            class_scores = model(feature_batch)
            loss = loss_function(class_scores, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(feature_batch)

        training_loss = float(total_loss.cpu()) / len(dataset)
        if early_stopping:
            validation_predictions = np.argmax(
                model.predict_proba(validation_features),
                axis=1,
            )
            score = f1_score(
                validation_targets,
                validation_predictions,
                labels=np.arange(len(model.classes_)),
                average="macro",
                zero_division=0,
            )
        else:
            score = -training_loss

        significantly_improved = score > best_score + tolerance
        if score > best_score:
            best_score = score
            best_epoch = epoch
            if early_stopping:
                best_state_dict = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

        if significantly_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "n_epochs": epoch,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score if early_stopping else None,
        "final_training_loss": training_loss,
    }



def _encode_labels(labels, class_labels):
    label_to_index = {
        int(label): index
        for index, label in enumerate(class_labels)
    }
    try:
        return np.asarray(
            [label_to_index[int(label)] for label in labels],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            f"Dataset contains label {error.args[0]} that is not configured"
        ) from error


def _plot_failure_cases(data, predictions, class_names_by_label, plot_config):
    if not bool(plot_config.get("enabled", False)):
        return []
    if "sample_index" not in predictions.columns:
        print("Skipping failure plots because predictions lack sample_index")
        return []

    from src.data_proc.plot_tools import plot_vdf_xz_slice
    from src.data_proc.vdf_helpers import get_vdf_plot_parameters_from_file

    splits = _resolve_failure_plot_splits(
        plot_config.get("splits", ["validation", "test"])
    )
    max_per_pair = int(plot_config.get("max_per_pair", 8))
    if max_per_pair <= 0:
        raise ValueError("failure_plots.max_per_pair must be positive")

    vdflim = float(plot_config.get("vdflim", 2e6))
    failures = predictions[
        (~predictions["correct"])
        & predictions["split"].isin(splits)
    ]
    if failures.empty:
        return []

    output_paths = []
    plot_parameter_cache = {}
    vdf_shape = tuple(data["X"].shape[1:])

    group_columns = ["split", "true_label", "predicted_label"]
    for (split, true_label, predicted_label), group in failures.groupby(
        group_columns,
        sort=True,
    ):
        true_label = int(true_label)
        predicted_label = int(predicted_label)
        true_class_name = class_names_by_label[true_label]
        predicted_class_name = class_names_by_label[predicted_label]

        for _, failure in group.head(max_per_pair).iterrows():
            sample_index = int(failure["sample_index"])
            metadata_row = data["metadata"].iloc[sample_index].to_dict()
            file_location = metadata_row["file_location"]
            cid = int(metadata_row["cid"])
            cache_key = (file_location, cid, vdf_shape)

            if cache_key not in plot_parameter_cache:
                plot_parameter_cache[cache_key] = get_vdf_plot_parameters_from_file(
                    file_location=file_location,
                    cid=cid,
                    vdf_shape=vdf_shape,
                )
            extent, dv, threshold = plot_parameter_cache[cache_key]

            output_path = (
                data["output_dir"]
                / "failure_plots"
                / str(split)
                / f"true_{true_class_name}"
                / f"pred_{predicted_class_name}"
                / _create_failure_plot_filename(metadata_row, sample_index)
            )
            plot_vdf_xz_slice(
                vdf=data["X"][sample_index],
                y_label=true_label,
                metadata_row=metadata_row,
                extent=extent,
                output_path=output_path,
                dv=dv,
                threshold=threshold,
                vdflim=vdflim,
                predicted_class_name=predicted_class_name,
            )
            output_paths.append(output_path)

    return output_paths


def _resolve_failure_plot_splits(configured_splits):
    if isinstance(configured_splits, str):
        return [configured_splits]
    return [str(split) for split in configured_splits]


def _create_failure_plot_filename(metadata_row, sample_index):
    timestep = metadata_row.get("timestep", "unknown")
    cid = metadata_row.get("cid", "unknown")
    return f"sample_{sample_index:06d}_t{timestep}_cid{cid}.png"


def _resolve_class_weight(configured_class_weight):
    if configured_class_weight is None:
        return "none"

    class_weight = str(configured_class_weight).lower()
    if class_weight in {"false", "no", "none", "unweighted"}:
        return "none"
    if class_weight in {"soft", "sqrt", "sqrt_balanced", "soft_balanced"}:
        return "sqrt_balanced"
    if class_weight in {"true", "yes", "balanced", "weight", "weights", "weighted"}:
        return "balanced"
    if class_weight in {"manual", "custom"}:
        return "manual"

    raise ValueError(
        "class_weight must be 'none', 'sqrt_balanced', 'balanced', or 'manual'"
    )


def _create_class_weights(
    targets,
    n_classes,
    class_weight,
    class_names,
    configured_class_weights,
):
    if class_weight == "none":
        return None

    class_counts = np.bincount(targets, minlength=n_classes).astype(float)
    if np.any(class_counts == 0.0):
        raise ValueError("Cannot create class weights for empty classes")

    if class_weight == "manual":
        weights = _create_manual_class_weight_array(
            class_names=class_names,
            configured_class_weights=configured_class_weights,
            config_name="model.class_weights",
        )
    else:
        weights = len(targets) / (n_classes * class_counts)
        if class_weight == "sqrt_balanced":
            weights = np.sqrt(weights)

    return weights.astype(np.float32)


def _create_training_sampler(targets, class_names, sampler_config, random_seed):
    sampler_config = sampler_config or {}
    enabled = bool(sampler_config.get("enabled", False))
    mode = str(sampler_config.get("mode", "none")).lower()
    if not enabled or mode in {"none", "false", "off", "disabled"}:
        return None, {
            "enabled": False,
            "mode": "none",
            "class_weights": None,
        }

    if mode not in {"balanced", "sqrt_balanced", "manual"}:
        raise ValueError(
            "model.sampler.mode must be 'balanced', 'sqrt_balanced', or 'manual'"
        )

    class_counts = np.bincount(targets, minlength=len(class_names)).astype(float)
    if np.any(class_counts == 0.0):
        raise ValueError("Cannot create sampler weights for empty classes")

    if mode == "manual":
        class_weights = _create_manual_class_weight_array(
            class_names=class_names,
            configured_class_weights=sampler_config.get("class_weights", {}),
            config_name="model.sampler.class_weights",
        )
    else:
        class_weights = len(targets) / (len(class_names) * class_counts)
        if mode == "sqrt_balanced":
            class_weights = np.sqrt(class_weights)

    sample_weights = class_weights[np.asarray(targets, dtype=int)]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(targets),
        replacement=bool(sampler_config.get("replacement", True)),
        generator=torch.Generator().manual_seed(random_seed),
    )
    return sampler, {
        "enabled": True,
        "mode": mode,
        "class_weights": class_weights.astype(np.float32),
    }


def _create_manual_class_weight_array(
    class_names,
    configured_class_weights,
    config_name,
):
    if not configured_class_weights:
        raise ValueError(f"{config_name} must be set when using manual weights")

    configured_class_weights = {
        str(class_name): float(weight)
        for class_name, weight in configured_class_weights.items()
    }
    missing_classes = [
        class_name
        for class_name in class_names
        if class_name not in configured_class_weights
    ]
    if missing_classes:
        raise ValueError(
            f"{config_name} missing weights for classes: {missing_classes}"
        )

    weights = np.asarray(
        [configured_class_weights[class_name] for class_name in class_names],
        dtype=float,
    )
    if np.any(weights <= 0.0):
        raise ValueError(f"{config_name} values must be positive")

    return weights


def _resolve_batch_size(configured_batch_size, n_samples):
    if configured_batch_size == "auto":
        return min(32, n_samples)
    batch_size = min(int(configured_batch_size), n_samples)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive or 'auto'")
    return batch_size


def _resolve_prediction_batch_size(configured_batch_size, model_batch_size, n_samples):
    prediction_batch_size = _resolve_batch_size(configured_batch_size, n_samples)
    return min(prediction_batch_size, model_batch_size)


def _resolve_adaptive_pool_shape(configured_shape, ndim=2):
    if configured_shape is None:
        configured_shape = [4] * ndim

    if isinstance(configured_shape, str):
        values = [int(value.strip()) for value in configured_shape.split(",")]
    else:
        values = [int(value) for value in configured_shape]

    if len(values) != ndim or any(value <= 0 for value in values):
        raise ValueError(
            f"adaptive_pool_shape must contain {ndim} positive integers"
        )

    return tuple(values)


def _resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _set_random_seed(random_seed, deterministic):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
