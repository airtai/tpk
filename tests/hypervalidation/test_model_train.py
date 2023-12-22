import pytest

from tpk.hypervalidation import train_model
from tpk.torch import TSMixerModel


@pytest.mark.slow
def test_train_model() -> None:
    wrmsse = train_model(
        model_cls=TSMixerModel,
        data_path="data/m5",
        batch_size=64,
        epochs=1,
        context_length=20,
        n_block=1,
        hidden_size=64,
        weight_decay=0.01,
        dropout_rate=0.01,
        disable_future_feature=False,
        lr=0.001,
        use_static_feat=True,
        use_one_cycle=False,
    )
    assert wrmsse > 0.0
    assert wrmsse < 100.0
