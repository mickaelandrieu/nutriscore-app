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

## Train the model

1. Download the RAW data on OpenFoodFacts ;
2. Execute `notebooks/eda.ipynb` notebook to clean the data ;
3. Execute `notebooks/feature_encoding_and_selection.ipynb` notebook to transform the data ;
4. Execute `src/create_folds.py` to prepare data for training ;
5. Execute `src/train.py` to train the model ;
5. Execute `streamlit run src/app.py` to launch the Streamlit app ;
6. (Soon) Execute `src/api.py` to get predictions from a REST endpoint ;

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
python src/report.py --fold=1
```

> fold value is in range [0,4]

![report (10th of January, 2022)](https://user-images.githubusercontent.com/1247388/148713711-50f92ccb-6e59-44bf-9e6c-558e86e9a9be.JPG)

## Test the machine learning model

### Using a Streamlit Web app

```bash
streamlit run src/app.py
```

## LICENSE

This project is provided under the MIT license.
