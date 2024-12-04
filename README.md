# Parkinson-Desease-Prediction-Model-Using-Machine-Learning


## Overview
This project leverages machine learning algorithms to predict the likelihood of Parkinson's disease based on various biomedical voice measurements. 

## Objectives
- To build a machine learning model for predicting Parkinson's disease.                  
- To evaluate the performance of different algorithms on the dataset.
- To explore data balancing techniques for improved model accuracy.
- To integrate insights from Generative AI and Prompt Engineering for better usability and interface design.

## Key Features
- **Data Preprocessing**: Includes techniques for handling imbalanced datasets using upsampling.
- **Model Comparison**: Compares the performance of models such as Random Forest, SVM, KNN, and others on both original and balanced datasets.
- **Performance Metrics**: Provides detailed accuracy scores, classification reports, and other metrics for each model.

## Project Structure
The project is organized into the following sections:

1. **Data Analysis and Preprocessing**  
   - Loading and inspecting the Parkinson's dataset.  
   - Handling missing and duplicate values.  
   - Oversampling for dataset balancing.

2. **Model Training and Evaluation**  
   - Implementing various machine learning algorithms.  
   - Comparing results before and after dataset balancing.  
   - Selecting the best-performing model.  
   - Key results:  
     - Best accuracy (Initial Dataset): **KNN - 92.3%**  
     - Best accuracy (Balanced Dataset): **Random Forest - 94.4%**


3. **Graphs and Visualizations**  
   - Accuracy comparison graphs for different models.  
   - Workflow diagram for data preprocessing and model evaluation.  
   - Feature importance visualization (for Random Forest).

## Requirements
- Python 3.8+
- Libraries:  
  - `pandas`, `numpy`  
  - `scikit-learn`  
  - `matplotlib`, `seaborn` (for visualizations)  
  - `joblib`, `pickle` (for model serialization)


