# README

## Machine Learning Mastery with Python

This repository has the notebooks for prectice code and notes, taken from
Machine Learning Mastery with python by [Jason Brawnlee](https://machinelearningmastery.com/about/).
This book focues on a specific sub-field of machine learning called predictive modeling, 
which is the most useful in industry and thge type of machine learning that the scikit-learn library in Python excels.

![cover](https://github.com/Bluelord/ML_Mastery_Python/blob/bfe671661ac544fa963948408374f4b011022782/Images/ML_Mastery_python.JPG)

### **Lessons** are divied into 6 sections

  - **Define Preblem:** Investigate and characterize the problem for better understanding.
  0. Python Ecosystem for Machine Learning.
  1. Python and SciPy Crash Course.
  2. Load Datasets from CSV.
 
  - **Analyze Data:** Discriptive Statistics & visualization of Data.
  3. Understand Data With Descriptive Statistics. (Analyze Data)
  4. Understand Data With Visualization. (Analyze Data)

  - **Prepare Data:** Data transformation for better exposer for struturing to the modeling algorithm.
  5. Pre-Process Data. (Prepare Data)
  6. Feature Selection. (Prepare Data)

  - **Evaluate Algorithms:** Design a test harness for different stadard algorithms & select the top few to investigate.
  7. Resampling Methods. (Evaluate Algorithms)
  8. Algorithm Evaluation Metrics. (Evaluate Algorithms)
  9. Spot-Check Classification Algorithms. (Evaluate Algorithms)
  10. Spot-Check Regression Algorithms. (Evaluate Algorithms)
  11. Model Selection. (Evaluate Algorithms)
  12. Pipelines. (Evaluate Algorithms)
  
  - **Improve Results:** Use algorithm tuning & ensemble models.
  13. Ensemble Methods. (Improve Results)
  14. Algorithm Parameter Tuning. (Improve Results)
  
  - **Present Result:** Finalize the model, make prediction & present results.
  15. Model Finalization. (Present Results)

From this book we will know how to work on small to medium size dataset, 
how to build a modle that can make accurate predictions on new data, 
we wil learn new and different techniques in python and scipy.

### Python Ecosystem for ML

Python is a general purpose interpreted programming language. It is easy to learn and use
primarily because the language focuses on readability.

SciPy is an ecosystem of Python libraries for mathematics, science and engineering.
The SciPy ecosystem is comprised of the following core modules relevant to machine learning:
- NumPy: A foundation for SciPy that allows you to efficiently work with data in arrays.
- Matplotlib: Allows you to create 2D charts and plots from data.
- Pandas: Tools and data structures to organize and analyze your data.

The scikit-learn library is how you can develop and practice machine learning in Python.
The focus of the library is machine learning algorithms for classification, regression,
clustering and more. It also provides tools for related tasks such as evaluating models, tuning
parameters and pre-processing data.

#### **Python Ecosystem Installations**

```
# scipy
import scipy
print('scipy: %s' % scipy.__version__)

# numpy
import numpy
print('numpy: %s' % numpy.__version__)

# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)

# pandas
import pandas
print('pandas: %s' % pandas.__version__)

# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
```
