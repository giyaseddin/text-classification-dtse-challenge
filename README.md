Multi-label Text Classification
==============================

Experiments and serving for Multi-label Text Classification Model

## Table of Contents
- [Project Overview](#project-overview)
- [Installation and Setup](#installation-and-setup)
- [Project Structure](#project-structure)
- [Workflow and Experiments](#workflow-and-experiments)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [Contact Information](#contact-information)

## Project Overview
## Project Overview

### Objective
The objective of this project is to develop a robust multi-label text classification model capable of accurately categorizing text into three distinct categories: cyber security, environmental issues, and others. This model aims to assist in automatically sorting texts based on their content, enhancing efficiency in content management and monitoring.

### Background
This project sits at the intersection of Natural Language Processing (NLP) and machine learning, focusing on the challenge of classifying text data into multiple categories. Multi-label classification is a complex area in NLP because each text can belong to more than one category, differing from traditional single-label classification problems. 

The task is further complicated by common issues in text data, such as class imbalance where some categories are significantly underrepresented compared to others, ambiguous or overlapping class definitions, and the high dimensionality of textual data. Addressing these challenges requires careful design of data preprocessing, feature engineering, and the selection of appropriate modeling techniques that can effectively handle the nuances of multi-label text datasets.

## Installation and Setup
- Dependencies: Different tools were used on the way of exploring the best model to train, so, libraries like: `scikit-learn`, `lightgbm`, `transformers` are included explicitly or implicitly in the `requirements.txt` file
- Installation: It is enough to run the following command to install
```shell
make requirements
```
or if there's a different environment choice, then a regular pip install will work:
```shell
pip install -U pip setuptools wheel

pip3 install -r requirements.txt
``` 
- Execution: To run any of the experiments, after the installation, and activating the right environment, then running the jupyter notebook:

```shell
jupyter notebook
```
this will open the window of the project in the browser, and find the related project folders there. Then you can find ipython notebooks under `notebooks`.


## Workflow and Experiments
The development of the classification models involved a series of structured steps, each building upon the insights and results of the previous:

- **Exploratory Data Analysis (EDA)**:
  - Performed initial EDA to understand data characteristics, trends, and challenges.
  - Identified issues such as class imbalance and homogeneity in the data. Outliers were also handled at this stage.
  - Relevant insights and trends are detailed in the respective EDA notebook.

- **Data Cleaning**:
  - Removed noisy and unnecessary data that could potentially impact learning algorithms negatively.
  - Ensured the cleaning process did not affect the overall data accuracy.

- **Data Splitting**:
  - Executed a canonical split into training, validation, and test sets, preparing them for model training.
  - Additionally, created an upsampled version of the training set to address class imbalance.

- **Initial Model Training with Scikit-Learn**:
  - Started with simple, fast-to-implement sklearn models to gauge the difficulty of the problem.
  - Employed SVC with tf-idf vectorization, achieving an overall macro F1 score of approximately 0.54.
  - Experimented with both single model and per-class model approaches, with maximum per-class F1 scores reaching up to 0.76.

- **Advanced Classifier with Gradient Boosting**:
  - Transitioned to gradient boosting classifiers using LightGBM, improving performance to 0.60 overall macro F1 score and approximately 0.77 macro F1 score per class, still using tf-idf vectorization.

- **Implementation of Sentence Transformers**:
  - Shifted vector representation to sentence transformers for more contextually aware embeddings.
  - Utilized `BAAI/bge-small-en-v1.5` embedding model from the MTEB leaderboard with sklearn's SVC, resulting in F1 macro scores of approximately 0.80 for both classes.
  - LightGBM with `BAAI/bge-small-en-v1.5` embeddings achieved F1 macro scores of 0.75 (cyber) and 0.84 (environment).

- **Fine-tuning BERT for Multi-label Classification**:
  - Fine-tuned a BERT model with a classification head specifically designed for multi-label tasks.
  - Achieved F1 macro scores of 0.71 and 0.81 for the individual classes and about 0.68 overall.

- **SetFit for Sentence Transformers**:
  - Employed SetFit to fine-tune sentence transformers using a contrastive learning method, reaching F1 scores of 0.74 and 0.81 for the classes.

- **Testing and Validation**:
  - Utilized the `classification_report` from Scikit-Learn to provide detailed insights into the performance of each model across different metrics such as precision, recall, and F1 score.
  - While overall accuracy is a commonly referenced metric, it does not always provide a complete picture, especially in the context of class imbalances and multi-label classification. Therefore, a more granular and per-class look at metrics like F1 score and recall is essential. These metrics offer a deeper understanding of model performance, particularly in how well the model handles each individual class.
  - These enhanced metrics were critical for assessing the effectiveness of the models under conditions where class representation varies significantly, ensuring that the evaluation reflected both the overall and class-specific accuracy and robustness of the models.

### Reflections on Findings
- The progression of model complexity and the shift from traditional vectorization to deep learning embeddings markedly improved classification performance.
- Although hyperparameter tuning was manually conducted based on prior experience, automating this process might offer marginal gains. Nonetheless, manual adjustments provided significant enhancements in model performance.
- The series of experiments underscore the importance of contextually rich representations and sophisticated model architectures in tackling multi-label classification challenges effectively.
- The use of sentence transformers, particularly with the `BAAI/bge-small-en-v1.5` embedding model, consistently demonstrated the highest F1 macro scores, underscoring the effectiveness of advanced embedding techniques in improving the classification accuracy.

## Results
The results are presented as macro averages of precision, recall, and F1-score for each model and experiment. Here's a detailed breakdown:
### Cyber Security Class Results Table
| Notebook/Experiment                          | Precision | Recall | F1-Score |
|----------------------------------------------|-----------|--------|----------|
| **initial-exploratory-experiment.ipynb**     |           |        |          |
| Exp1                                         | 0.73      | 0.76   | 0.74     |
| Exp2                                         | 0.74      | 0.70   | 0.72     |
| **sklearn-sentence-transformer.ipynb**       |           |        |          |
| Exp1                                         | 0.77      | **0.83**| **0.80** |
| Exp2                                         | 0.76      | **0.83**| 0.79     |
| **lightgbm-sentence-transformer.ipynb**      |           |        |          |
| Exp1                                         | **0.83**  | 0.70   | 0.75     |
| **lightgbm-tfidf.ipynb**                     |           |        |          |
| Exp1                                         | 0.80      | 0.75   | 0.78     |
| **setfit-text-classification_multilabel_full.ipynb** |    |        |          |
| Exp1                                         | 0.73      | 0.75   | 0.74     |
| **BERT-ft.ipynb**                            |           |        |          |
| Exp1                                         | 0.78      | 0.78   | 0.78     |

### Environmental Issue Class Results Table
| Notebook/Experiment                          | Precision | Recall | F1-Score |
|----------------------------------------------|-----------|--------|----------|
| **initial-exploratory-experiment.ipynb**     |           |        |          |
| Exp1                                         | 0.77      | 0.76   | 0.76     |
| Exp2                                         | 0.75      | 0.73   | 0.74     |
| **sklearn-sentence-transformer.ipynb**       |           |        |          |
| Exp1                                         | 0.80      | **0.84**   | 0.81     |
| Exp2                                         | 0.78      | 0.83   | 0.80     |
| **lightgbm-sentence-transformer.ipynb**      |           |        |          |
| Exp1                                         | **0.86**  | 0.82   | **0.84** |
| **lightgbm-tfidf.ipynb**                     |           |        |          |
| Exp1                                         | 0.78      | 0.75   | 0.76     |
| **setfit-text-classification_multilabel_full.ipynb** |    |        |          |
| Exp1                                         | 0.80      | 0.82   | 0.81     |
| **BERT-ft.ipynb**                            |           |        |          |
| Exp1                                         | 0.80      | 0.83   | 0.81     |


This tabulated data provides a clear and concise comparison of model performances across different experiments, highlighting the effectiveness of different approaches in tackling the complexities of multi-label text classification.

The results demonstrate that models employing sentence transformers generally outperform those using tf-idf vectorization, with notable improvements in both precision and recall, particularly for environmental issues.

**Note:** There are more experiments done, but failing ones weren't reported or shared.


### Test set
The predictions are extracted from the best performing model (according to the table above) and can be found and downloaded from here [test set predictions](reports/best_model_test_preds.csv)

## Discussion
The experiments reveal that advanced NLP techniques, especially those using sentence transformers and BERT fine-tuning, consistently deliver superior F1 scores. This underscores their capacity to effectively manage the complexities associated with multi-label text classification. Notably, the LightGBM model augmented with sentence transformer embeddings proved especially effective, achieving the highest F1 score of 0.84 for environmental issues. These findings highlight the crucial role that cutting-edge NLP technologies play in addressing common challenges such as class imbalance and in boosting the overall performance of classification models.


## Conclusion and Future Work

### Conclusion
> The project demonstrated the effectiveness of advanced NLP techniques, particularly sentence transformers and BERT fine-tuning, in enhancing multi-label text classification. The LightGBM model using sentence transformer embeddings performed exceptionally well, especially in classifying environmental issues.

### Future Work
Efforts to further improve the model will focus on the following technical objectives:

- [ ] **Explore Advanced Ensemble Methods**: Combine different models to leverage their strengths and improve accuracy.
- [ ] **Implement AUC-ROC Analysis**: Use AUC-ROC curves to assess model performance in distinguishing between classes. Set different thresholds if that works well with a certain model.
- [ ] **Conduct Detailed Error Analysis**: Identify and address specific weaknesses in the model.
- [ ] **Evaluate Feature Importance**: Analyze which features most significantly impact model predictions.
- [ ] **Expand Cross-Validation**: Ensure the model’s effectiveness across various data segments.
- [ ] **Test Additional Deep Learning Architectures**: Experiment with newer architectures to compare improvements.
- [ ] **Utilize Data Augmentation**: Increase training data variety to enhance model robustness.
- [ ] **Integrate Multi-Modal Data**: Incorporate different data types for a more comprehensive analysis.
- [ ] **Apply Domain Adaptation Techniques**: Adapt the model to perform well across varying data distributions.

These steps are designed to refine the model’s predictive capabilities and ensure its practical applicability across different scenarios.



## Project Structure

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
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
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── reporting  <- Scripts to create exploratory and results oriented reporting
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
