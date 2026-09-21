# Kaggle Competition Project

This repository contains my solution and experiments for a **Kaggle machine learning competition** (https://www.kaggle.com/competitions/ukraine-ml-bootcamp-2023).
It serves as a workspace for experimenting with data preprocessing, feature engineering, model training, evaluation, and generation of competition submissions.

## Project Goals
The goal of the competition is to identify yoga pose based on the provided image.
The main goals of the project are to:

* explore and understand the competition dataset;
* preprocess and prepare the data for machine learning models;
* experiment with different features and modelling approaches;
* evaluate model performance;
* improve the solution iteratively;
* generate predictions in the format required for Kaggle submission.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/alexpolishchuck/kaggle_competition.git
cd kaggle_competition
```

It is recommended to create a virtual environment before running the project:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

or on Windows:

```bash
.venv\Scripts\activate
```

Install the Python packages required by the scripts or notebooks in `kaggle_comp/`.

## Workflow

The general workflow of the project is:

```text
Raw Data
   ↓
Data Exploration
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Validation / Evaluation
   ↓
Prediction
   ↓
Kaggle Submission
```

Experiments can be modified and rerun to compare different preprocessing strategies, feature sets, models, and hyperparameters.

## Data
This dataset consists of a train.csv file which contains the image_id and class_6 columns. The image_id is the unique id for each image of yoga pose and the class_6 classes of yoga poses to which that image belongs. The images for training and testing are present in the images folder. The aim of this competition is to predict the class_6 correctly for the test_images and submit the response as csv file submission.csv.
