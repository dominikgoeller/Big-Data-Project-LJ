# Big-Data-Project-LJ
Final project for the course Big Data at University Ljubljana  

We are following the CRISP-DM methodology for conducting a data mining project. 

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Business Understanding
tbd

## Data Understanding
1) Data Description: Parking Violations Issuance datasets contain violations issued during the respective fiscal year. The Issuance datasets are not updated to reflect violation status, the information only represents the violation(s) at the time they are issued. Since appearing on an issuance dataset, a violation may have been paid, dismissed via a hearing, statutorily expired, or had other changes to its status. To see the current status of outstanding parking violations, please look at the Open Parking & Camera Violations dataset.
2) Data Exploration Report: 14.4M rows and 43 columns. 
3) Data Quality Report:

## Data Preparation
1) Select Data: Which columns are important for our task etc
2) Clean Data: Only if necessary. Probably useful to remove rows with NaN, 0, blank or wrong(dates in the future e.g.) values.
3) Merge/Augment Data: With  
  a) Weather Information,  
  b) Vicinity/Location of primary and high schools  
  c) Information about events in vicinity  
  d) Vicinity/Location of major businesses  
  e) Vicinity/Location of major attractions  
4) Format Data: Only if necessary.

## Modeling
tbd
## notes
hypothesen aufstellen
weather data for districts
business areas definieren (> 50 businesses in einer area) und nehmen das dann als ganzes
spatial map for school start and school end for parking ticket violations in a certain area around schools
correlation of events in may vs parking tickets (maybe example events like fashion week and pride parade etc)

## Evaluation
tbd

## Deployment
tbd
