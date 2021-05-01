# README

## Projects

Python_ML_Practice
These are my self-learning ML practice projects using Python(sklearn) following the instruction of "Machine Learning Mastery With Python BY Jason Brownlee". Gnerally there are two parts in this repository. The first one is mainly about ML basics such as loading file and describing data. In this README file, basics exercises are on Pima Indians onset of diabetes dataset. The second part is about three comprehensive ML projects on some well-known dataset using the basic knowledge in the first part. The details can be found using links below.

Classification Problem - Iris Flower Dataset
Regression Problem - Boston Housing Price Dataset
Binary Classification Problem - Sonar, Mines and Rocks Dataset
Getting Started with Basics
Overall, when given a new ML project, the workflow could be as following:

Define problem
Investigate and characterize the problem and clarify the project goal.

Summarize data
Use descriptive statistics and visualization techniques to get a grasp of data.

Descriptive Statistics
data dimension, type, attribute features (count, mean, std, min/max, percentiles), class categories, correlations between attributes, skew of univariate distributions

Visualization
univariate plots(histograms, density plot, boxplot), multivariate plots(correlation matrix plot, scatter plot matrix)

Data preprocessing [Incompleted]

Transformation
The reason for preprocessing data is that different algorithms make different assumptions about data requiring different transformation. Here are some common processing techniques:

Rescaling
To limit varying attributes ranges all between 0 and 1. Useful for weight-inputs regression/neural networks and kNN.          

Standardization
    To transform attributes with a Gaussian distribution to a standard Gaussian distribution (0 mean and 1 std). Useful for linear/logistic regression and LDA

Normalization
To rescaling each observation (row) to have a length of 1 (called a unit norm or a vector with the length of 1 in linear algebra). Useful for sparse dataset with varying attribute scales, weight-input neural network and kNN

Binarization
To transform data using a binary threshold.(1 for above threshold and 0 for below threshold)

Feature Selection
Irrelevant or partially relevant features can negatively impact model performance, such as decreasing the accuracy of many models. Feature Selection is to select features that contribute most to the prediction variable or output in which you are interested. It can help reduce overfiting, improve accuracy, reduce training time. Here are some common processing techniques:

Statistical Test Selection with chi-2
    To select those features that have the strongest relationship with output variable

Recursive Feature Elimination (RFE)
  To recursively removing attributes and building a model on those attributes that remain.

Principal Component Analysis (PCA)
A kind of data reduction technique. It uses linear algebra to transform the dataset into a compressed form and choose the number of dimensions or principal components in the transformed result.

Feature importance
To use bagged decision trees such as Random Forest and Extra Trees to estimate the importance of features.

Algorithm Evaluation

Separate train/test dataset (Resampling)
In most cases, k-fold Cross Validation technique (e.g. k = 3, 5 or 10) will be used to estimate algorithm performance with less variance. At first, the dataset will be splited into k parts. Then the algorithm is trained on k-1 folds with one held back and tested on the held back fold. Repeatedly, each fold of the dataset will be given a chance to be the held back test set. After all these, you can summarize using the mean and std of such k different performance scores.

Performance Metrics
Choice of metrics influence how the performance of ML algorithms is measure and compared, as it represents how you weight the importance of different characteristics in the output results and ultimate choice of which algorithm to choose.

For Classification Problem

Classification Accuracy
Classification Accuracy is the ratio of the number of correct predictions and the numberof all predictions. Only suitable for equal number of obsevations in each class, and all predictions and prediction errors are equally important.

Logorithmic Loss
Logorithmic Loss is to evaluate the predictions of probabilities of membership to a given class. Corrected or incorrected prediction errors will be rewarded/punished proportionally to the comfidence of the prediction. Smaller logloss is better with 0 for a perfect logloss.

Area Under ROC Curve (AUC)
AUC is used to evaluate binary classification problem, representing a model's ability to discriminate between positive and negative classes. (1 for perfect prediction. 0.5 for as good as random.)
         ROC can be broken down into sensitivity (true positive rate) and specificity (true negative rate)

Confusion Matrix
Confusion Matrix is representation of models' classes accuracy. Generally, the majority of the predictions fall on the diagonal line of the matrix.

Classification Report
A scikit-learn lib provided classification report including precision, recall and F1-score.

For Regression Problem

Mean Absolute Error (MAE) L1-norm
MAE is the sum of the absolute differences between predictions and actual values.

Mean Squared Error (MSE) L2-norm
MSE is the sum of square root of the mean squared error

R^2
R^2 is an indication of the goodness of fit of a set of predictions to the actual values range between 0 (non-fit) and 1 (perfiect fit). Statistically, it is called coefficient of determination

Spot-Checking Algorithm
Spot-Checking is a way to discover which algs perform well on ML problem. Since we do not know which algs will work best on our dataset, we can make a guess and further dig deeper. Here are some common algs:

Classification

Logistic Regression (Linear)
Linear Discriminant Analysis (Linear)
k-Nearest Neighbors (Non-linear)
Naive Bayes (Non-linear)
Classification and Regression (Non-linear)
Support Vector Machine (Non-linear)
Regression

Linear Regression (Linear)
Ridge Regression (Linear)
LASSO Linear Regression (Linear)
Elastic Net Regression (Linear)
Naive Bayes (Non-linear)
Classification and Regression (Non-linear)
Support Vector Machine (Non-linear)
Improve results

Ensemble
Ensemble learning helps improve machine learning results by combining several models. This approach allows the production of better predictive performance compared to a single model.

Bagging
Bagging tries to implement similar learners on small sample populations and then takes a mean of all the predictions.

Boosting
Boosting is an iterative technique which adjust the weight of an observation based on the last classification. If an observation was classified incorrectly, it tries to increase the weight of this observation and vice versa. Boosting in general decreases the bias error and builds strong predictive models. However, they may sometimes over fit on the training data.

Stacking
Here we use a learner to combine output from different learners. This can lead to decrease in either bias or variance error depending on the combining learner we use.

Params Tuning
Algs tuning is the final step in AML before finalizing model. In scikit-learn, there are two simple methods for params tuning:

Grid Search Param Tuning
Methodically build and evaluate a model for each combination of algorithm parameters specified in a grid

Random Search Param Tuning
Sample algorithm parameters from a random distribution (i.e. uniform) for a fixed number of iterations. A model is constructed and evaluated for each combination of parameters chosen.

Present results
The final part includes:

Predictions on validation dataset
Create standalone model on entire training dataset
Save model for later use
