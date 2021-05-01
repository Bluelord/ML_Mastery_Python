# README

## Projects

In this Machine Learning Mastery with Python practice projects are presented using Python(sklearn)
following the instruction given in this book by Jason Brownlee. After completing this project
lesson we will get to know, how to structure an end-to-end predictive modeling project,
how to map the tasks you have learned till now onto a project, 
how to best use the structured project template to ensure an accurate result for your dataset.

### Machine Learning [Project Template](https://github.com/Bluelord/ML_Mastery_Python/blob/4a47aa14dd168a7feeede187fb8b0932b1c70b81/Projects/Project%20Template.py) Steps
Overall, when given a new ML project, the workflow could be as following:

1. **Prepare problem**
    - Investigate and characterize the problem and clarify the project goal.
    - Python modules, classes and functions that you intend to use.
    - Loading your dataset from CSV.

2. **Summarize data** <br/>
Use descriptive statistics and visualization techniques to get a grasp of data.
    - Descriptive Statistics <br/>
    Data dimension, type, attribute features (count, mean, std, min/max, percentiles), 
    class categories, correlations between attributes, skew of univariate distributions.
    - Visualization <br/>
    Univariate plots(histograms, density plot, boxplot), multivariate plots(correlation matrix plot, scatter plot matrix)

3. **Prepare Data** (preprocessing) <br/>
    - Transformation or Cleaning data by handling duplicates & missing values. <br/>
The reason for preprocessing data is that different algorithms make different assumptions about data requiring different transformation. Here are some common processing techniques:
        - Rescaling <br/>
To limit varying attributes ranges all between 0 and 1. Useful for weight-inputs regression/neural networks and kNN.
        - Standardization <br/>
    To transform attributes with a Gaussian distribution to a standard Gaussian distribution (0 mean and 1 std). Useful for linear/logistic regression and LDA
        - Normalization <br/>
To rescaling each observation (row) to have a length of 1 (called a unit norm or a vector with the length of 1 in linear algebra). Useful for sparse dataset with varying attribute scales, weight-input neural network and kNN.
        - Binarization <br/>
To transform data using a binary threshold.(1 for above threshold and 0 for below threshold)
    - Feature Selection <br/>
Irrelevant or partially relevant features can negatively impact model performance, 
such as decreasing the accuracy of many models. Feature Selection is to select features 
that contribute most to the prediction variable or output in which you are interested.
It can help reduce overfiting, improve accuracy, reduce training time. Here are some common processing techniques:
        - Statistical Test Selection with chi-2 <br/>
    To select those features that have the strongest relationship with output variable
        - Recursive Feature Elimination (RFE) <br/>
  To recursively removing attributes and building a model on those attributes that remain.
        - Principal Component Analysis (PCA) <br/>
A kind of data reduction technique. It uses linear algebra to transform the dataset into a compressed form and choose the number of dimensions or principal components in the transformed result.
        - Feature importance<br/>
To use bagged decision trees such as Random Forest and Extra Trees to estimate the importance of features.
4. **Algorithm Evaluation**<br/>
    - Separate train/test dataset (Resampling)<br/>
In most cases, k-fold Cross Validation technique (e.g. k = 3, 5 or 10) will be used to estimate algorithm performance with less variance. At first, the dataset will be splited into k parts. Then the algorithm is trained on k-1 folds with one held back and tested on the held back fold. Repeatedly, each fold of the dataset will be given a chance to be the held back test set. After all these, you can summarize using the mean and std of such k different performance scores.
    - Performance Metrics For Classification Problem <br/>
        - Classification Accuracy <br/>
Classification Accuracy is the ratio of the number of correct predictions and the numberof all predictions. Only suitable for equal number of obsevations in each class, and all predictions and prediction errors are equally important.
        - Logorithmic Loss<br/>
Logorithmic Loss is to evaluate the predictions of probabilities of membership to a given class. Corrected or incorrected prediction errors will be rewarded/punished proportionally to the comfidence of the prediction. Smaller logloss is better with 0 for a perfect logloss.
        - Area Under ROC Curve (AUC)<br/>
AUC is used to evaluate binary classification problem, representing a model's ability to discriminate between positive and negative classes. (1 for perfect prediction. 0.5 for as good as random.) ROC can be broken down into sensitivity (true positive rate) and specificity (true negative rate)
        - Confusion Matrix <br/>
Confusion Matrix is representation of models' classes accuracy. Generally, the majority of the predictions fall on the diagonal line of the matrix.
        - Classification Report<br/>
A scikit-learn lib provided classification report including precision, recall and F1-score.
    - Performance MetricsFor Regression Problem<br/>
        - Mean Absolute Error (MAE) L1-norm <br/>
MAE is the sum of the absolute differences between predictions and actual values.
        - Mean Squared Error (MSE) L2-norm <br/>
MSE is the sum of square root of the mean squared error
        - R^2 <br/>
R^2 is an indication of the goodness of fit of a set of predictions to the actual values range between 0 (non-fit) and 1 (perfiect fit). Statistically, it is called coefficient of determination
    - Spot-Checking Algorithm for Classification <br/>
        - Logistic Regression (Linear)
        - Linear Discriminant Analysis (Linear)
        - k-Nearest Neighbors (Non-linear)
        - Naive Bayes (Non-linear)
        - Classification and Regression (Non-linear)
        - Support Vector Machine (Non-linear)
    - Spot-Checking Algorithm for Regression
        - Linear Regression (Linear)
        - Ridge Regression (Linear)
        - LASSO Linear Regression (Linear)
        - Elastic Net Regression (Linear)
        - Naive Bayes (Non-linear)
        - Classification and Regression (Non-linear)
        - Support Vector Machine (Non-linear)
5. **Improve results**<br/>
    - Ensemble<br/>
Ensemble learning helps improve machine learning results by combining several models. This approach allows the production of better predictive performance compared to a single model.
        - Bagging<br/>
Bagging tries to implement similar learners on small sample populations and then takes a mean of all the predictions.
        - Boosting<br/>
Boosting is an iterative technique which adjust the weight of an observation based on the last classification. If an observation was classified incorrectly, it tries to increase the weight of this observation and vice versa. Boosting in general decreases the bias error and builds strong predictive models. However, they may sometimes over fit on the training data.
        - Stacking<br/>
Here we use a learner to combine output from different learners. This can lead to decrease in either bias or variance error depending on the combining learner we use.
    - Parameters Tuning <br/>
Algs tuning is the final step in AML before finalizing model. In scikit-learn, there are two simple methods for params tuning:
        - Grid Search Param Tuning<br/>
Methodically build and evaluate a model for each combination of algorithm parameters specified in a grid
        - Random Search Param Tuning<br/>
Sample algorithm parameters from a random distribution (i.e. uniform) for a fixed number of iterations. A model is constructed and evaluated for each combination of parameters chosen.
6. Present results<br/>
The final part includes:
    - Predictions on validation dataset
    - Create standalone model on entire training dataset
    - Save model for later use
