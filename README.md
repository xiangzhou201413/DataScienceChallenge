# DataScienceChallenge

# Takeaways from initial data exploration:
    The data are mixed with category and numerical types, and are unbalenced with only 22% of default labels (Y = 1).
# Implement any data pre-processing steps, if required
    One hot encoding is used to transform category data, sample downsizing is used to balance the data to prevent prediction bias and false accuracies. 
# Explore analytical techniques or models that are appropriate for this use case
    Several classifies implentmented for testing, including XGBoost, AdaBoost, Random Forest, Ridge, and neural network
# Fit selected model to the analytics ready dataset. List outcomes from model fitting
    For one of the runs, 
    Accuracy for Xgboost: 83.2% 
    Accuracy for Ridge: 80.7% 
    Accuracy for AdaBoost: 82.7% 
    Accuracy for Neural network: 63.5% 
    Accuracy for Random F: 82.0
# Evaluate model performance on held-out/testing dataset
    For selected model of XGboost, use grid search to find the best parameters and 5 fold corss validation to get scores:
    Best Parameters:
    5 120
    cross validation scores:
    [0.81694915 0.80112994 0.8089676  0.82780708 0.83383572]
# What other options would you’ve considered (in solving the problem), if you had additional time?
    I would try to test some other classifiers and further to combine them to produce better results. 
