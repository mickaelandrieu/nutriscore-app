# OpenFoodFacts Nutriscore predictor

## How to get the dataset

As the dataset is too big to be shared on GitHub, you must download it on [OpenFoodFacts (products.csv: 4Go)](https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv).

## Local installation

```bash
python -m venv dev
source dev/Scripts/activate
pip install -r requirements.txt
```

## Docker installation

### Build the image

```bash
docker build --tag app:1.0 .
```

### Access the Streamlit application

```bash
docker run --publish 8501:8501 -it app:1.0 -m streamlit run src/app.py
```

Then access [http://localhost:8501](http://localhost:8501).

### Access the REST API

```bash
docker run --publish 8501:8501 -it app:1.0 src/api.py
```

Then access [http://localhost:8501](http://localhost:8501/docs)

> Everytime you update the project, you must build a new image with a new tag.

## Train the model

1. Download the RAW data on OpenFoodFacts ;
2. Execute `notebooks/eda.ipynb` notebook to clean the data ;
3. Execute `notebooks/feature_encoding_and_selection.ipynb` notebook to transform the data ;
4. Execute `src/create_folds.py` to prepare data for training ;
5. Execute `src/train.py` to train the model ;

## Evaluate the performance of the model

```bash
python src/report.py --fold=1
```

> fold value is in range [0,4]

![report (10th of January, 2022)](https://user-images.githubusercontent.com/1247388/148713711-50f92ccb-6e59-44bf-9e6c-558e86e9a9be.JPG)

## Test the machine learning model (Without Docker)

### Using a Streamlit Web app

```bash
streamlit run src/app.py
```

### Using a REST API

```bash
uvicorn src.api:app --reload
```

## Quality tools

```bash
python -m isort src/
python -m black src/
python -m flake8 src/ --count --statistics
```

## LICENSE

This project is provided under the MIT license.
