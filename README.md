# OpenFoodFacts Nutriscore predictor

## How to get the dataset

As the dataset is too big to be shared on GitHub, you must download it on [OpenFoodFacts (products.csv: 4Go)](https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv).

## Installation

```bash
python -m venv dev
source dev/Scripts/activate
pip install -r requirements.txt
```

> A Docker file will be provided soon.

## Usage

TODO : provide streamlit app and REST endpoint for prediction (we must commit the model.bin).

## Train the model

1. Download the RAW data on OpenFoodFacts ;
2. Execute `notebooks/eda.ipynb` notebook to clean the data ;
3. Execute `notebooks/feature_encoding_and_selection.ipynb` notebook to transform the data ;
4. Execute `src/create_folds.py` to prepare data for training ;
5. Execute `src/train.py` to train the model ;
5. (Soon) Execute `src/app.py` to launch the streamlit app ;
6. (Sonn) Execute `src/api.py` to get predictions from a REST endpoint ;

## Quality tools

```bash
python -m isort src/
```

```bash
python -m black src/
```

```bash
python -m flake8 src/ --count --statistics
```

## Evaluate the performance of the model

```bash
python src/report.py
```

![report](https://user-images.githubusercontent.com/1247388/148670889-39e3b0c7-c07d-421e-90b2-94a504549d59.JPG)


## LICENSE

This project is provided under the MIT license.
