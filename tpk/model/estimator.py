import logging
from typing import Any, Dict, Iterator, List, Optional

import lightning.pytorch as pl
import torch
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.env import env
from gluonts.itertools import Cached, Cyclic, IterableSlice, PseudoShuffled
from gluonts.model.forecast_generator import (
    DistributionForecastGenerator,
)
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.model.estimator import PyTorchLightningEstimator, TrainOutput
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler
from lightning import LightningModule
from lightning.pytorch.tuner.tuning import Tuner
from torch.utils.data import DataLoader

from .lightning_module import TSMixerLightningModule
from .module import TSMixerModel

logger = logging.getLogger(__name__)

PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class IterableDataset(torch.utils.data.IterableDataset):  # type: ignore
    def __init__(self, iterable: Iterator[Any]):
        self.iterable = iterable

    def __iter__(self) -> Any:
        yield from self.iterable


class TSMixerEstimator(PyTorchLightningEstimator):  # type: ignore
    """
    Estimator class to train a TSMixer model.

    This class is uses the model defined in ``TSMixerModel``, and wraps it
    into a ``TSMixerLightningModule`` for training purposes: training is
    performed using PyTorch Lightning's ``pl.Trainer`` class.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict.
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length).
    n_block
        Number of TSMixer blocks (default: 2).
    hidden_size
        Number of hidden size for each layer (default: 128).
    weight_decay
        Weight decay regularization parameter (default: ``1e-8``).
    dropout_rate
        Dropout regularization parameter (default: 0.1).
    patience
        Patience parameter for learning rate scheduler.
    num_feat_dynamic_real
        Number of dynamic real features in the data (default: 0).
    num_feat_static_real
        Number of static real features in the data (default: 0).
    num_feat_static_cat
        Number of static categorical features in the data (default: 0).
    cardinality
        Number of values of each categorical feature.
        This must be set if ``num_feat_static_cat > 0`` (default: None).
    embedding_dimension
        Dimension of the embeddings for categorical features
        (default: ``[min(50, (cat+1)//2) for cat in cardinality]``).
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).
    loss
        Loss to be optimized during training
        (default: ``NegativeLogLikelihood()``).
    scaling
        Whether to automatically scale the target values (default: true).
    time_features
        List of time features, from :py:mod:`gluonts.time_feature`, to use as
        inputs of the RNN in addition to the provided data (default: None,
        in which case these are automatically determined based on freq).
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch
        Number of batches to be processed in each training epoch
        (default: 50).
    trainer_kwargs
        Additional arguments to provide to ``pl.Trainer`` for construction.
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    """

    @validated()  # type: ignore
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        epochs: int,
        context_length: Optional[int] = None,
        n_block: int = 2,
        hidden_size: int = 128,
        weight_decay: float = 1e-8,
        dropout_rate: float = 0.1,
        patience: int = 10,
        num_feat_dynamic_real: int = 0,
        disable_future_feature: bool = False,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: Optional[DistributionOutput] = None,
        loss: Optional[DistributionLoss] = None,
        scaling: bool = True,
        time_features: Optional[List[TimeFeature]] = None,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.epochs = epochs
        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.patience = patience
        self.distr_output = StudentTOutput() if distr_output is None else distr_output
        self.loss = NegativeLogLikelihood() if loss is None else loss
        self.n_block = n_block
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.disable_future_feature = disable_future_feature
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = (
            cardinality if cardinality and num_feat_static_cat > 0 else [1]
        )
        self.embedding_dimension = embedding_dimension
        self.scaling = scaling
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                if not self.num_feat_static_cat > 0
                else []
            )
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
                if not self.num_feat_static_real > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension
                    expected_ndim=1 + len(self.distr_output.event_shape),
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.num_feat_dynamic_real > 0
                        else []
                    ),
                ),
            ]
        )

    def _create_instance_splitter(
        self, module: LightningModule, mode: str
    ) -> InstanceSplitter:
        if mode not in ["training", "validation", "test"]:
            raise RuntimeError("mode must be either training, validation or test")

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: LightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        transformation = self._create_instance_splitter(
            module, "training"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(data), shuffle_buffer_length=shuffle_buffer_length
            )
        )

        return IterableSlice(
            iter(
                # nosemgrep
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=self.batch_size,
                    num_workers=2,
                    persistent_workers=True,
                    **kwargs,
                )
            ),
            self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: LightningModule,
        **kwargs: Any,
    ) -> DataLoader:  # type: ignore
        transformation = self._create_instance_splitter(
            module, "validation"
        ) + SelectFields(TRAINING_INPUT_NAMES)

        validation_instances = transformation.apply(data)

        # nosemgrep
        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            num_workers=2,
            persistent_workers=True,
            **kwargs,
        )

    def create_lightning_module(self) -> LightningModule:
        model = TSMixerModel(
            freq=self.freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_feat_dynamic_real=(
                self.num_feat_dynamic_real + len(self.time_features)
            ),
            num_future_feat=(
                # len(self.time_features)
                0
                if self.disable_future_feature
                else self.num_feat_dynamic_real + len(self.time_features)
            ),
            num_feat_static_real=max(1, self.num_feat_static_real),
            num_feat_static_cat=max(1, self.num_feat_static_cat),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            n_block=self.n_block,
            hidden_size=self.hidden_size,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            scaling=self.scaling,
        )

        return TSMixerLightningModule(  # type: ignore
            model=model,
            loss=self.loss,
            weight_decay=self.weight_decay,
            patience=self.patience,
            epochs=self.epochs,
            steps_per_epoch=self.num_batches_per_epoch,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: LightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            forecast_generator=DistributionForecastGenerator(self.distr_output),
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        from_predictor: Optional[PyTorchPredictor] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> TrainOutput:
        transformation = self.create_transformation()

        with env._let(max_idle_transforms=max(len(training_data), 100)):
            transformed_training_data: Dataset = transformation.apply(
                training_data, is_train=True
            )
            if cache_data:
                transformed_training_data = Cached(transformed_training_data)

            training_network = self.create_lightning_module()

            training_data_loader = self.create_training_data_loader(
                transformed_training_data,
                training_network,
                shuffle_buffer_length=shuffle_buffer_length,
            )

        validation_data_loader = None

        if validation_data is not None:
            with env._let(max_idle_transforms=max(len(validation_data), 100)):
                transformed_validation_data: Dataset = transformation.apply(
                    validation_data, is_train=True
                )
                if cache_data:
                    transformed_validation_data = Cached(transformed_validation_data)

                validation_data_loader = self.create_validation_data_loader(
                    transformed_validation_data,
                    training_network,
                )

        if from_predictor is not None:
            training_network.load_state_dict(from_predictor.network.state_dict())

        monitor = "train_loss" if validation_data is None else "val_loss"
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor, mode="min", verbose=True
        )

        custom_callbacks = self.trainer_kwargs.pop("callbacks", [])
        trainer = pl.Trainer(
            **{
                "accelerator": "auto",
                "callbacks": [checkpoint] + custom_callbacks,
                **self.trainer_kwargs,
            }
        )

        tuner = Tuner(trainer)

        tuner.lr_find(
            model=training_network,
            train_dataloaders=training_data_loader,
            val_dataloaders=validation_data_loader,
            early_stop_threshold=50.0,
        )

        trainer.fit(
            model=training_network,
            train_dataloaders=training_data_loader,
            val_dataloaders=validation_data_loader,
            ckpt_path=ckpt_path,
        )

        if checkpoint.best_model_path != "":
            logger.info(f"Loading best model from {checkpoint.best_model_path}")
            best_model = training_network.__class__.load_from_checkpoint(
                checkpoint.best_model_path
            )
        else:
            best_model = training_network

        return TrainOutput(
            transformation=transformation,
            trained_net=best_model,
            trainer=trainer,
            predictor=self.create_predictor(transformation, best_model),
        )
