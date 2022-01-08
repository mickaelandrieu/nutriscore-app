from setuptools import setup, find_packages

setup(
    name='nutriscore-app',
    version='0.1.0',
    description='OpenFoodFacts Nutriscore predictor',
    author='Mickaël Andrieu',
    author_email='mickael.andrieu@solvolabs.com',
    url='https://github.com/mickaelandrieu/nutriscore-app/setup-py',
    packages=find_packages(),
     install_requires=[
        'matplotlib==3.4.3',
        'numpy==1.21.2',
        'pandas==1.3.4',
        'plotly==5.3.1',
        'scikit-learn==1.0',
        'scipy==1.7.1',
        'streamlit==1.1.0',
        'joblib==1.1.0',
        'black==21.12b0',
        'flake8==4.0.1',
        'flake8-black==0.2.3',
        'flake8-requirements==1.5.1',
        'flake8-nb==0.3.1',
        'pandas-vet==0.2.2',
        'rich==10.16.2',
        'wemake-python-styleguide==0.16.0'
    ],
    setup_requires=['flake8'],  
)