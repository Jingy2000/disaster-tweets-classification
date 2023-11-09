# Disaster Tweets Classification with BERT

This project aims to classify tweets into disaster-related and non-disaster-related categories. I trained RNNs and fine-tuned a BERT model to achieve this task.

## Table of Contents

- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Project Structure

```
.
|-- clean data.ipynb: Jupyter notebook for data exploring, cleaning and preprocessing.
|-- data: Directory containing the raw and cleaned datasets.
|-- main.py: Main script to run the model.
|-- model: Directory to store trained model checkpoints.
|-- results: Directory containing submission files.
|-- runs: Tensorboard logs for different runs.
`-- src: Source code directory.
    |-- config.py: Configuration parameters for the model.
    |-- dataset.py: Dataset preparation and loading utilities.
    |-- model.py: BERT, RNN model definition.
    |-- train.py: Training loop and utilities.
```

## Methodology

### Data Preprocessing

In the notebook, I performed various preprocessing steps to make the data suitable for a language model and explore the dataset. This includes:

- Handling missing values
- Removing special characters and URLs, expanding contractions
- Reduced the unknown word rate in embeddings from 45% to 4%.

These preprocessing efforts significantly improve the data quality, paving the way for a robust language model training.

### Training Strategy

I employed a combination of:

- Monitoring validation loss to explore the hyperparameters, ensuring the model doesn't overfit and  fine-tuning model hyperparameters
- Using AdamW optimizer, learning rate schedules with warm-up steps.
- Using pre-trained Glove and Word2Vec embeddings to do transfer learning for word embedding.
- Ensuring efficient RNN training by implementing padding and masking for batch sequences.

## Results

Our fine-tuned BERT model achieved an F1 score of 0.833 on the test set.

## Acknowledgements

- Dataset source: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data)
