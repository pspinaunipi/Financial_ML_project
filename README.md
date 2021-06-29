# Bagging and boosting in financial machine learning
[![Documentation Status](https://readthedocs.org/projects/financial-ml-project/badge/?version=latest)](https://financial-ml-project.readthedocs.io/en/latest/?badge=latest)

![header](https://nastglobal.com/wp-content/uploads/2019/10/cognitive-technology-finance.jpg)

Welcome to our project repository! Here you will find all the codes,images and tables we created for the [JaneStreet kaggle competition](https://www.kaggle.com/c/jane-street-market-prediction/overview)


# Table of contents


- [Bagging and boosting in financial machine learning](bagging-and-boosting-in-financial-machine-learning)
- [Table of contents](#table-of-contents)
- [Overview](#overview)
- [Import and clean dataset](#import-and-clean-dataset)
- [Grid and Random search](#grid-and-random-search)
- [Analyze the results](#analyze-the-results)
- [Documentation](#documentation)

# Overview
Our main task were:
1) Import clean and visualize the competition dataset.
2) Select a classifier and find the best hyperparameters through random and grid searches with cross validation.
3) Analyze the results of the searches to find the best classifier.


[(Back to top)](#table-of-contents)


# Import and clean dataset
The codes we wrote for this task are:
1) initial_import.py: [[code](https://github.com/pspinaunipi/Financial_ML_project/blob/main/main/initial_import.py), [documentation](https://financial-ml-project.readthedocs.io/en/latest/start/initial_import.html)] <br>In this code we define the fuctions to import and clean the dataset
2) introduction.py: [[code](https://github.com/pspinaunipi/Financial_ML_project/blob/main/main/introduction.py), [documentation](https://financial-ml-project.readthedocs.io/en/latest/start/introduction.html)] <br>This is the first code we wrote to visualize the dataset
3) data_visualization_main.py [[code](https://github.com/pspinaunipi/Financial_ML_project/blob/main/main/data_visualization_main.py), [documentation](https://financial-ml-project.readthedocs.io/en/latest/start/data_visualization_main.html)] <br> This is the second code we wrote to visualize the dataset

 
[(Back to top)](#table-of-contents)

<!-- This is optional and it is used to give the user info on how to use the project after installation. This could be added in the Installation section also. -->

# Grid and random search
The main code for this task is rf_search.py [[code](https://github.com/pspinaunipi/Financial_ML_project/blob/main/main/rf_search.py), [documentation](https://financial-ml-project.readthedocs.io/en/latest/search/search.html)] <br>


[(Back to top)](#table-of-contents)

# Analyze the results
The main codes for this task are:
1) analyze_results_adaboost [[code](https://github.com/pspinaunipi/Financial_ML_project/blob/main/main/analyze_results_adaboost.py)]
2) analyze_results_bayeisan [[code](https://github.com/pspinaunipi/Financial_ML_project/blob/main/main/analyze_results_bayesian.py)]
3) analyze_results_forest [[code](https://github.com/pspinaunipi/Financial_ML_project/blob/main/main/analyze_results_forest.py)]
4) analyze_results_xgboost[[code](https://github.com/pspinaunipi/Financial_ML_project/blob/main/main/analyze_results_xgboost.py)]
5) scoring classifiers [[code](https://github.com/pspinaunipi/Financial_ML_project/blob/main/main/scoring_classifiers.py), [documentation](https://financial-ml-project.readthedocs.io/en/latest/analyze_results/scoring_classifiers.html)] <br>In this code we compare different classifiers using the same cross validation method

[(Back to top)](#table-of-contents)

# Documentation
To see the full documentation click the badge near the title or click [here](https://financial-ml-project.readthedocs.io/en/latest/index.html)

[(Back to top)](#table-of-contents)
<!-- Let's also add a footer because I love footers and also you **can** use this to convey important info.

Let's make it an image because by now you have realised that multimedia in images == cool(*please notice the subtle programming joke). -->


<!-- Add the footer here -->

<!-- ![Footer](https://github.com/navendu-pottekkat/awesome-readme/blob/master/fooooooter.png) -->
