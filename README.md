# Disaster Tweets Classification with BERT

This project aims to classify tweets into disaster-related and non-disaster-related categories. We fine-tuned a BERT model to achieve this task.

## Table of Contents

- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Project Structure

```
.
|-- clean data.ipynb: Jupyter notebook for data cleaning and preprocessing.
|-- data: Directory containing the raw and cleaned datasets.
|-- main.py: Main script to run the model.
|-- model: Directory to store trained model checkpoints.
|-- results: Directory containing submission files.
|-- runs: Tensorboard logs for different runs.
`-- src: Source code directory.
    |-- config.py: Configuration parameters for the model.
    |-- dataset.py: Dataset preparation and loading utilities.
    |-- model.py: BERT model definition.
    |-- train.py: Training loop and utilities.
```

## Methodology

### Data Preprocessing

In the notebook, I performed various preprocessing steps to make the data suitable for a BERT model. This includes:

- Handling missing values
- Removing special characters and URLs, expanding contractions

### Training Strategy

I employed a combination of:

- Monitoring validation loss to explore the hyperparameters, ensuring the model doesn't overfit
- Using AdamW optimizer, learning rate schedules with warm-up steps, 

## Results

Our fine-tuned BERT model achieved an F1 score of 0.833 on the test set.

## Acknowledgements

- Dataset source: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data)
