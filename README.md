# tpk (Temporal Predictions Kit)

A collection of tools, models and functionalities for hanling timeseries datasets



<!-- <img class="hide-on-website" height="100px" src="https://ts.gluon.ai/dev/_static/gluonts.svg"> -->

<!-- # GluonTS - Probabilistic Time Series Modeling in Python -->

<p align="center">
  <a href="https://github.com/airtai/tpk/actions/workflows/test.yaml" target="_blank">
    <img src="https://github.com/airtai/tpk/actions/workflows/test.yaml/badge.svg?branch=main" alt="Test Passing"/>
  </a>

  <a href="https://coverage-badge.samuelcolvin.workers.dev/redirect/airtai/tpk" target="_blank">
      <img src="https://coverage-badge.samuelcolvin.workers.dev/airtai/tpk.svg" alt="Coverage">
  </a>

  <a href="https://www.pepy.tech/projects/tpk" target="_blank">
    <img src="https://static.pepy.tech/personalized-badge/tpk?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month" alt="Downloads"/>
  </a>

  <a href="https://pypi.org/project/tpk" target="_blank">
    <img src="https://img.shields.io/pypi/v/tpk?label=PyPI" alt="Package version">
  </a>

  <a href="https://pypi.org/project/tpk" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/tpk.svg" alt="Supported Python versions">
  </a>

  <br/>

  <a href="https://github.com/airtai/tpk/actions/workflows/codeql.yml" target="_blank">
    <img src="https://github.com/airtai/tpk/actions/workflows/codeql.yml/badge.svg" alt="CodeQL">
  </a>

  <a href="https://github.com/airtai/tpk/actions/workflows/dependency-review.yaml" target="_blank">
    <img src="https://github.com/airtai/tpk/actions/workflows/dependency-review.yaml/badge.svg" alt="Dependency Review">
  </a>

  <a href="https://github.com/airtai/tpk/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/airtai/tpk.png" alt="License">
  </a>

  <a href="https://github.com/airtai/tpk/blob/main/CODE_OF_CONDUCT.md" target="_blank">
    <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Code of Conduct">
  </a>

  <a href="https://discord.gg/qFm6aSqq59" target="_blank">
      <img alt="Discord" src="https://img.shields.io/discord/1085457301214855171?logo=discord">
  </a>
</p>


<!-- GluonTS is a Python package for probabilistic time series modeling, focusing on deep learning based models,
based on [PyTorch](https://pytorch.org) and [MXNet](https://mxnet.apache.org). -->


## Installation

**tpk** requires Python 3.7 or newer, and the easiest way to install it is via
`pip`:

```bash
pip install tpk"
```

<!-- See the [documentation](https://ts.gluon.ai/stable/getting_started/install.html)
for more info on how GluonTS can be installed. -->

## Simple Example

To illustrate how to use tpk, we train a model and make predictions
using the airpassengers dataset. The dataset consists of a single time
series of monthly passenger numbers between 1949 and 1960. We train the model
on the first nine years and make predictions for the remaining three years.

```py
import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator
from tpk.torch import TPKEstimator

# Load data from a CSV file into a PandasDataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/"
    "TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv",
    index_col=0,
    parse_dates=True,
)
dataset = PandasDataset(df, target="#Passengers")

# Split the data for training and testing
training_data, test_gen = split(dataset, offset=-36)
test_data = test_gen.generate_instances(prediction_length=12, windows=3)

# Train the model and make predictions
model = TPKEstimator(
    prediction_length=12, freq="M", trainer_kwargs={"max_epochs": 5}
).train(training_data)

forecasts = list(model.predict(test_data.input))

# Plot predictions
plt.plot(df["1954":], color="black")
for forecast in forecasts:
  forecast.plot()
plt.legend(["True values"], loc="upper left", fontsize="xx-large")
plt.show()
```

##### todo: replace me
![[train-test]](https://ts.gluon.ai/static/README/forecasts.png)

Note, the forecasts are displayed in terms of a probability distribution and
the shaded areas represent the 50% and 90% prediction intervals.


## Contributing

If you wish to contribute to the project, please refer to our
[contribution guidelines](https://github.com/airt/tpk/tree/dev/CONTRIBUTING.md).

## Citing

If you use tpk in a scientific publication, we encourage you to add the following references to the related papers,
in addition to any model-specific references that are relevant for your work:

<!-- ```bibtex
@article{gluonts_jmlr,
  author  = {Alexander Alexandrov and Konstantinos Benidis and Michael Bohlke-Schneider
    and Valentin Flunkert and Jan Gasthaus and Tim Januschowski and Danielle C. Maddix
    and Syama Rangapuram and David Salinas and Jasper Schulz and Lorenzo Stella and
    Ali Caner Türkmen and Yuyang Wang},
  title   = {{GluonTS: Probabilistic and Neural Time Series Modeling in Python}},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {116},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v21/19-820.html}
}
```

```bibtex
@article{gluonts_arxiv,
  author  = {Alexandrov, A. and Benidis, K. and Bohlke-Schneider, M. and
    Flunkert, V. and Gasthaus, J. and Januschowski, T. and Maddix, D. C.
    and Rangapuram, S. and Salinas, D. and Schulz, J. and Stella, L. and
    Türkmen, A. C. and Wang, Y.},
  title   = {{GluonTS: Probabilistic Time Series Modeling in Python}},
  journal = {arXiv preprint arXiv:1906.05264},
  year    = {2019}
}
``` -->

## Links

### Documentation

* [Documentation](https://tpk.airt.ai/)

<!-- ### References

* [JMLR MLOSS Paper](http://www.jmlr.org/papers/v21/19-820.html)
* [ArXiv Paper](https://arxiv.org/abs/1906.05264)
* [Collected Papers from the group behind GluonTS](https://github.com/awslabs/gluonts/tree/dev/REFERENCES.md): a bibliography. -->

<!-- ### Tutorials and Workshops

* [Tutorial at IJCAI 2021 (with videos)](https://lovvge.github.io/Forecasting-Tutorial-IJCAI-2021/) with [YouTube link](https://youtu.be/AB3I9pdT46c).
* [Tutorial at WWW 2020 (with videos)](https://lovvge.github.io/Forecasting-Tutorial-WWW-2020/)
* [Tutorial at SIGMOD 2019](https://lovvge.github.io/Forecasting-Tutorials/SIGMOD-2019/)
* [Tutorial at KDD 2019](https://lovvge.github.io/Forecasting-Tutorial-KDD-2019/)
* [Tutorial at VLDB 2018](https://lovvge.github.io/Forecasting-Tutorial-VLDB-2018/)
* [Neural Time Series with GluonTS](https://youtu.be/beEJMIt9xJ8)
* [International Symposium of Forecasting: Deep Learning for Forecasting workshop](https://lostella.github.io/ISF-2020-Deep-Learning-Workshop/) -->

## Stay in touch

Please show your support and stay in touch by:

- giving our [GitHub repository](https://github.com/airtai/tpk/){.external-link target="_blank"} a star, and

- joining our [Discord server](https://discord.gg/qFm6aSqq59){.external-link target="_blank"}

Your support helps us to stay in touch with you and encourages us to
continue developing and improving the framework. Thank you for your
support!

---

## Contributors

Thanks to all of these amazing people who made the project better!

<a href="https://github.com/airtai/tpk/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=airtai/tpk"/>
</a>
