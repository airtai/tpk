import pytest

from tpk.hypervalidation import train_model


@pytest.mark.slow
def test_train_model() -> None:
    wrmsse = train_model(
        data_path="data/m5",
        batch_size=64,
        epochs=1,
        context_length=20,
        n_block=1,
        hidden_size=64,
        weight_decay=0.01,
        dropout_rate=0.01,
        disable_future_feature=False,
        use_static_feat=True,
    )
    assert wrmsse > 0.0
    assert wrmsse < 100.0
