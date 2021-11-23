# title: "practical_exercise_8 , Methods 3, 2021, autumn semester"
# author: Juli Furjes
# date: 17/11/2021
# output: pdf_document


#%% Exercises and objectives

# 1) Load the magnetoencephalographic recordings and do some initial plots to understand the data  
# 2) Do logistic regression to classify pairs of PAS-ratings  
# 3) Do a Support Vector Machine Classification on all four PAS-ratings  

# REMEMBER: In your report, make sure to include code that can reproduce the answers requested in the exercises below (__MAKE A KNITTED VERSION__)  
# REMEMBER: This is Assignment 3 and will be part of your final portfolio   

#%% EXERCISE 1 - Load the magnetoencephalographic recordings and do some initial plots to understand the data  

# The files `megmag_data.npy` and `pas_vector.npy` can be downloaded here (http://laumollerandersen.org/data_methods_3/megmag_data.npy) and here (http://laumollerandersen.org/data_methods_3/pas_vector.npy)   

#%%% Exercise 1.1
# 1) Load `megmag_data.npy` and call it `data` using `np.load`. You can use `join`, which can be imported from `os.path`, to create paths from different string segments  
#     i. The data is a 3-dimensional array. The first dimension is number of repetitions of a visual stimulus , the second dimension is the number of sensors that record magnetic fields (in Tesla) that stem from neurons activating in the brain, and the third dimension is the number of time samples. How many repetitions, sensors and time samples are there?  
#     ii. The time range is from (and including) -200 ms to (and including) 800 ms with a sample recorded every 4 ms. At time 0, the visual stimulus was briefly presented. Create a 1-dimensional array called `times` that represents this.  
#     iii. Create the sensor covariance matrix $\Sigma_{XX}$: $$\Sigma_{XX} = \frac 1 N \sum_{i=1}^N XX^T$$ $N$ is the number of repetitions and $X$ has $s$ rows and $t$ columns (sensors and time), thus the shape is $X_{s\times t}$. Do the sensors pick up independent signals? (Use `plt.imshow` to plot the sensor covariance matrix)  
#     iv. Make an average over the repetition dimension using `np.mean` - use the `axis` argument. (The resulting array should have two dimensions with time as the first and magnetic field as the second)  
#     v. Plot the magnetic field (based on the average) as it evolves over time for each of the sensors (a line for each) (time on the x-axis and magnetic field on the y-axis). Add a horizontal line at $y = 0$ and a vertical line at $x = 0$ using `plt.axvline` and `plt.axhline`  
#     vi. Find the maximal magnetic field in the average. Then use `np.argmax` and `np.unravel_index` to find the sensor that has the maximal magnetic field.  
#     vii. Plot the magnetic field for each of the repetitions (a line for each) for the sensor that has the maximal magnetic field. Highlight the time point with the maximal magnetic field in the average (as found in 1.1.v) using `plt.axvline`  
#     viii. Describe in your own words how the response found in the average is represented in the single repetitions. But do make sure to use the concepts _signal_ and _noise_ and comment on any differences on the range of values on the y-axis  

import numpy as np
from os.path import join
path = '/Users/julifurjes/Documents/uni/Methods 3/classes/data'
data = np.load(join(path, "megmag_data.npy"))

#%%%% Exercise 1.1.i
data.shape

#%%%% Exercise 1.1.ii
times = np.arange(-200, 804, 4)
print(times)

#%%%% Exercise 1.1.iii
import matplotlib.pyplot as plt
n = 682
covariance = []

for i in range(n):
  covariance.append(data[i,:,:] @ data[i,:,:].T)

covariance = sum(covariance)/n

plt.figure()
plt.imshow(covariance)
plt.show()

#%%%% Exercise 1.1.iv
repetition_mean = (np.mean(data, axis=0))
repetition_mean = repetition_mean.T
print(repetition_mean.shape)
print(repetition_mean.ndim)
print(repetition_mean)

#%%%% Exercise 1.1.v
plt.plot(times, repetition_mean)
plt.axvline(0)
plt.axhline(0)

#%%%% Exercise 1.1.vi
print(np.amax(repetition_mean))
print(np.unravel_index(np.argmax(repetition_mean), repetition_mean.shape)) #sensor number 73

#%%%% Exercise 1.1.vii
for i in range(251):
    plt.plot(times, data[i, 73, :])
    
plt.axvline(112)
plt.show()

#%%%% Exercise 1.1.viii

#%%% Exercise 1.2
# 2) Now load `pas_vector.npy` (call it `y`). PAS is the same as in Assignment 2, describing the clarity of the subjective experience the subject reported after seeing the briefly presented stimulus  
#     i. Which dimension in the `data` array does it have the same length as?  
#     ii. Now make four averages (As in Exercise 1.1.iv), one for each PAS rating, and plot the four time courses (one for each PAS rating) for the sensor found in Exercise 1.1.vi 
#     iii. Notice that there are two early peaks (measuring visual activity from the brain), one before 200 ms and one around 250 ms. Describe how the amplitudes of responses are related to the four PAS-scores. Does PAS 2 behave differently than expected?

y = np.load(join(path, "pas_vector.npy"))
#%%%% Exercise 1.2.i
print(y.shape)

#%%%% Exercise 1.2.ii
pas1 = []
pas2 = []
pas3 = []
pas4 = []
for i in range(682):
    if y[i] == 1:
      pas1.append(y[i])
    if y[i] == 2:
      pas2.append(y[i])
    if y[i] == 3:
      pas3.append(y[i])
    if y[i] == 4:
     pas4.append(y[i])
sensor73 = data[:,73,:]      
avgpas1 = np.mean(sensor73[pas1], axis = 0)
avgpas2 = np.mean(sensor73[pas2], axis = 0)
avgpas3 = np.mean(sensor73[pas3], axis = 0)
avgpas4 = np.mean(sensor73[pas4], axis = 0)
plt.figure()
plt.plot(times, avgpas1)
plt.plot(times, avgpas2)
plt.plot(times, avgpas3)
plt.plot(times, avgpas4)
plt.axvline()
plt.axhline()
plt.legend(['pas 1', 'pas 2', 'pas 3', 'pas 4'])
plt.show()

#%%%% Exercise 1.2.iii
#TODO
    
#%% EXERCISE 2 - Do logistic regression to classify pairs of PAS-ratings  

# 1) Now, we are going to do Logistic Regression with the aim of classifying the PAS-rating given by the subject  
#     i. We'll start with a binary problem - create a new array called `data_1_2` that only contains PAS responses 1 and 2. Similarly, create a `y_1_2` for the target vector  
#     ii. Scikit-learn expects our observations (`data_1_2`) to be in a 2d-array, which has samples (repetitions) on dimension 1 and features (predictor variables) on dimension 2. Our `data_1_2` is a three-dimensional array. Our strategy will be to collapse our two last dimensions (sensors and time) into one dimension, while keeping the first dimension as it is (repetitions). Use `np.reshape` to create a variable `X_1_2` that fulfils these criteria.  
#     iii. Import the `StandardScaler` and scale `X_1_2`  
#     iv. Do a standard `LogisticRegression` - can be imported from `sklearn.linear_model` - make sure there is no `penalty` applied  
#     v. Use the `score` method of `LogisticRegression` to find out how many labels were classified correctly. Are we overfitting? Besides the score, what would make you suspect that we are overfitting?  
#     vi. Now apply the _L1_ penalty instead - how many of the coefficients (`.coef_`) are non-zero after this?  
#     vii. Create a new reduced $X$ that only includes the non-zero coefficients - show the covariance of the non-zero features (two covariance matrices can be made; $X_{reduced}X_{reduced}^T$ or $X_{reduced}^TX_{reduced}$ (you choose the right one)) . Plot the covariance of the features using `plt.imshow`. Compared to the plot from 1.1.iii, do we see less covariance?  
# 2) Now, we are going to build better (more predictive) models by using cross-validation as an outcome measure    
#     i. Import `cross_val_score` and `StratifiedKFold` from `sklearn.model_selection`  
#     ii. To make sure that our training data sets are not biased to one target (PAS) or the other, create `y_1_2_equal`, which should have an equal number of each target. Create a similar `X_1_2_equal`. The function `equalize_targets_binary` in the code chunk associated with Exercise 2.2.ii can be used. Remember to scale `X_1_2_equal`!  
#     iii. Do cross-validation with 5 stratified folds doing standard `LogisticRegression` (See Exercise 2.1.iv)  
#     iv. Do L2-regularisation with the following `Cs=  [1e5, 1e1, 1e-5]`. Use the same kind of cross-validation as in Exercise 2.2.iii. In the best-scoring of these models, how many more/fewer predictions are correct (on average)?  
#     v. Instead of fitting a model on all `n_sensors * n_samples` features, fit  a logistic regression (same kind as in Exercise 2.2.iv (use the `C` that resulted in the best prediction)) for __each__ time sample and use the same cross-validation as in Exercise 2.2.iii. What are the time points where classification is best? Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)  
#     vi. Now do the same, but with L1 regression - set `C=1e-1` - what are the time points when classification is best? (make a plot)?  
#     vii. Finally, fit the same models as in Exercise 2.2.vi but now for `data_1_4` and `y_1_4` (create a data set and a target vector that only contains PAS responses 1 and 4). What are the time points when classification is best? Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)  
# 3) Is pairwise classification of subjective experience possible? Any surprises in the classification accuracies, i.e. how does the classification score fore PAS 1 vs 4 compare to the classification score for PAS 1 vs 2?  

#%%% Exercise 2.1

#%%%% Exercise 2.1.i
data_1_2 = np.array(pas1 + pas2)
print(data_1_2.shape)

#The shape of the data is supposed to be (214, 102, 251) but it is not - need to fix this??
#it's (214,) because this array only contains the value itself (1 or 2), but no data attached to it. somehow we have to filter our 'data' dataset based on this and then we'll have the right size. no idea how though.
y_1_2 = []

#%%%% Exercise 2.1.ii
X_1_2 = data_1_2.reshape(-1, 2)
#Is it supposed to look like this?
print(X_1_2)

#%%%% Exercise 2.1.iii
from sklearn.preprocessing import StandardScaler
X_1_2 = StandardScaler.fit_transform(X_1_2)
#Does not work, probably because the shape is not right..

#%%%% Exercise 2.1.iv

#%%%% Exercise 2.1.v

#%%%% Exercise 2.1.vi

#%%%% Exercise 2.1.vii

#%%% Exercise 2.2

#%%%% Exercise 2.2.i
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

#%%%% Exercise 2.2.ii
def equalize_targets_binary(data, y):
    np.random.seed(7)
    targets = np.unique(y) ## find the number of targets
    if len(targets) > 2:
        raise NameError("can't have more than two targets")
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target)) ## find the number of each target
        indices.append(np.where(y == target)[0]) ## find their indices
    min_count = np.min(counts)
    # randomly choose trials
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count,replace=False)
    
    # create the new data sets
    new_indices = np.concatenate((first_choice, second_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    
    return new_data, new_y

#%%%% Exercise 2.2.iii

#%%%% Exercise 2.2.iv

#%%%% Exercise 2.2.v

#%%%% Exercise 2.2.vi

#%%%% Exercise 2.2.vii

#%%% Exercise 2.3

#%% EXERCISE 3 - Do a Support Vector Machine Classification on all four PAS-ratings

# 1) Do a Support Vector Machine Classification  
#     i. First equalize the number of targets using the function associated with each PAS-rating using the function associated with Exercise 3.1.i  
#     ii. Run two classifiers, one with a linear kernel and one with a radial basis (other options should be left at their defaults) - the number of features is the number of sensors multiplied the number of samples. Which one is better predicting the category?
#     iii. Run the sample-by-sample analysis (similar to Exercise 2.2.v) with the best kernel (from Exercise 3.1.ii). Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)
#     iv. Is classification of subjective experience possible at around 200-250 ms?  
# 2) Finally, split the equalized data set (with all four ratings) into a training part and test part, where the test part if 30 % of the trials. Use `train_test_split` from `sklearn.model_selection`  
#     i. Use the kernel that resulted in the best classification in Exercise 3.1.ii and `fit`the training set and `predict` on the test set. This time your features are the number of sensors multiplied by the number of samples.  
#     ii. Create a _confusion matrix_. It is a 4x4 matrix. The row names and the column names are the PAS-scores. There will thus be 16 entries. The PAS1xPAS1 entry will be the number of actual PAS1, $y_{pas1}$ that were predicted as PAS1, $\hat y_{pas1}$. The PAS1xPAS2 entry will be the number of actual PAS1, $y_{pas1}$ that were predicted as PAS2, $\hat y_{pas2}$ and so on for the remaining 14 entries.  Plot the matrix
#     iii. Based on the confusion matrix, describe how ratings are misclassified and if that makes sense given that ratings should measure the strength/quality of the subjective experience. Is the classifier biased towards specific ratings?  
    
def equalize_targets(data, y):
    np.random.seed(7)
    targets = np.unique(y)
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target))
        indices.append(np.where(y == target)[0])
    min_count = np.min(counts)
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count, replace=False)
    third_choice = np.random.choice(indices[2], size=min_count, replace=False)
    fourth_choice = np.random.choice(indices[3], size=min_count, replace=False)
    
    new_indices = np.concatenate((first_choice, second_choice,
                                 third_choice, fourth_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    
    return new_data, new_y