# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a supervised machine learning classification model to predict whether a person earns more than $50K per year based on census demographic and employment-related features. The final model is a RandomForestClassifier implemented with scikit-learn. Categorical variables are transformed using one-hot encoding and the trained encoder is saved alongside the trained model for reuse during inference in the deployed API.

## Intended Use
The intended use of this model is for educational purposes as part of an end-to-end ML pipeline project. The model demonstrates data preprocessing, model training, model evaluation (including slice performance), and deployment of inference using a REST API with FastAPI. This model is not intended for real-world decision-making or any high-stakes applications such as hiring, credit, healthcare, insurance, or legal decisions.

## Training Data
The training dataset is the provided census dataset (`census.csv`) which includes a combination of continuous and categorical features such as age, education, workclass, marital status, occupation, relationship, race, sex, capital gain/loss, hours worked per week, and native country. The dataset is split into training and testing subsets using a train/test split. The training subset is used to fit the OneHotEncoder on categorical features and to train the classification model.

## Evaluation Data
The evaluation dataset is the held-out test subset created during the train/test split. Model predictions are generated on the processed test data using the fitted encoder and trained model. In addition to overall evaluation, the model is evaluated on categorical slices by calculating metrics for each unique value of each categorical feature. Slice metrics are saved to `slice_output.txt`.

## Metrics
The model is evaluated using the following classification metrics:
- Precision
- Recall
- F1 score (F-beta score with beta = 1)

Overall model performance on the held-out test dataset:
- Precision: 0.7419
- Recall: 0.6384
- F1: 0.6863

Performance across categorical slices is also computed to observe how model performance varies across subgroups (for example, different values in workclass, education, or sex). These slice-level metrics help identify cases where performance may be weaker for certain groups. Slice evaluation output is saved to `slice_output.txt`.

## Ethical Considerations
This dataset includes sensitive demographic attributes such as race, sex, and native country. Using these variables in a predictive model could amplify unfair bias and lead to discriminatory outcomes if used in real-world systems. This project includes evaluation on data slices to help detect performance differences across categories, but additional fairness testing and bias mitigation would be required before considering any real deployment.

## Caveats and Recommendations
This model was built for a course project and may not generalize to populations outside the dataset. Some categorical slice groups have small sample sizes, which can make their slice performance metrics unstable. Future improvements could include hyperparameter tuning, additional cross-validation, and fairness evaluations to assess demographic parity and reduce bias. The model should not be used in production high-impact decision systems without deeper evaluation and governance controls.
