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

print(data.shape)

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
plt.axvline(0, color="red")
plt.axhline(0, color="red")

#%%%% Exercise 1.1.vi

print(np.amax(repetition_mean))
print(np.unravel_index(np.argmax(repetition_mean), repetition_mean.shape)) #sensor number 73

#%%%% Exercise 1.1.vii

for i in range(682):
    plt.plot(times, data[i, 73, :])

plt.axvline(times[112], color= "red")
plt.show()

#%%%% Exercise 1.1.viii

#This plot shows all 682 repetitions at sensor 74 (indexed as 73). 
#We see this average in plot 1.1.v as the highest peak (red line). 
#   At this line though, it appears like there are less negative values and more positive values. 
#This plot is less clear. It seems there is a lot more noise in this plot compared to plot 1.1.v where the averages are taken. There we see the signal more clearly.

#%%% Exercise 1.2
# 2) Now load `pas_vector.npy` (call it `y`). PAS is the same as in Assignment 2, describing the clarity of the subjective experience the subject reported after seeing the briefly presented stimulus  
#     i. Which dimension in the `data` array does it have the same length as?  
#     ii. Now make four averages (As in Exercise 1.1.iv), one for each PAS rating, and plot the four time courses (one for each PAS rating) for the sensor found in Exercise 1.1.vi 
#     iii. Notice that there are two early peaks (measuring visual activity from the brain), one before 200 ms and one around 250 ms. Describe how the amplitudes of responses are related to the four PAS-scores. Does PAS 2 behave differently than expected?

y = np.load(join(path, "pas_vector.npy"))
#%%%% Exercise 1.2.i

print(y.shape)

#%%%% Exercise 1.2.ii

pas1 = np.where(y == 1)
pas2 = np.where(y == 2)
pas3 = np.where(y == 3)
pas4 = np.where(y == 4)

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
plt.axvline(color="black")
plt.axhline(color="black")
plt.legend(['pas 1', 'pas 2', 'pas 3', 'pas 4'])
plt.show()

#%%%% Exercise 1.2.iii

#we expect pas 2 and 3 to be the most regular ones, since they're the non-extreme values (not like pas 1 and 4)

#until about 250ms, people choose pas 2 the most, while after that they start to swift towards pas 3
    
#%% EXERCISE 2 - Do logistic regression to classify pairs of PAS-ratings

#%%% Exercise 2.1
# 1) Now, we are going to do Logistic Regression with the aim of classifying the PAS-rating given by the subject
#       i. We'll start with a binary problem - create a new array called `data_1_2` that only contains PAS responses 1 and 2.
#       ii. Scikit-learn expects our observations (`data_1_2`) to be in a 2d-array, which has samples (repetitions) on dimension 1 and features (predictor variables) on dimension 2. Our `data_1_2` is a three-dimensional array. Our strategy will be to collapse our two last dimensions (sensors and time) into one dimension, while keeping the first dimension as it is (repetitions). Use `np.reshape` to create a variable `X_1_2` that fulfils these criteria. Similarly, create a `y_1_2` for the target vector
#       iii. Import the `StandardScaler` and scale `X_1_2`
#       iv. Do a standard `LogisticRegression` - can be imported from `sklearn.linear_model` - make sure there is no `penalty` applied
#       v. Use the `score` method of `LogisticRegression` to find out how many labels were classified correctly. Are we overfitting? Besides the score, what would make you suspect that we are overfitting?
#     vii. Create a new reduced $X$ that only includes the non-zero coefficients - show the covariance of the non-zero features (two covariance matrices can be made; $X_{reduced}X_{reduced}^T$ or $X_{reduced}^TX_{reduced}$ (you choose the right one)) . Plot the covariance of the features using `plt.imshow`. Compared to the plot from 1.1.iii, do we see less covariance?

#%%%% Exercise 2.1.i

pas12 = np.argwhere((y == 1) | (y == 2))
data_1_2 = np.squeeze(data[pas12,:,:])

print(pas12)
print(data_1_2)

print(pas12.shape)
print(pas12.ndim)

y_1_2 = np.squeeze(y[pas12])
print(y_1_2)
print(len(y_1_2))

#%%%% Exercise 2.1.ii

X_1_2 = data_1_2.transpose(0,1,2).reshape(-1, data_1_2.shape[0])
#BUT the shape of it should be the other way around
#reshape instead of transpose for the solution?
print(X_1_2)

#%%%% Exercise 2.1.iii

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_1_2 = sc.fit_transform(X_1_2)

#%%%% Exercise 2.1.iv

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(penalty = 'none')
log_fit = regressor.fit(X_1_2, y_1_2)

log_fit.coef_

print(log_fit)

#%%%% Exercise 2.1.v

logscore = log_fit.score(X_1_2, y_1_2)
print(logscore)
#we are overfitting because we didn't split up the data to train and test

#%%%% Exercise 2.1.vi

from sklearn.linear_model import LogisticRegression
np.random.seed(7)
regressor = LogisticRegression(penalty = 'l1', solver = 'liblinear')
log_fit_1 = regressor.fit(X_1_2, y_1_2)

log_fit_1.coef_

np.count_nonzero(log_fit_1.coef_) #counting the non-zero coefficients


#%%%% Exercise 2.1.vii

#we see more covariance in this plot

#%%% Exercise 2.2
# 2) Now, we are going to build better (more predictive) models by using cross-validation as an outcome measure    
#     i. Import `cross_val_score` and `StratifiedKFold` from `sklearn.model_selection`  
#     ii. To make sure that our training data sets are not biased to one target (PAS) or the other, create `y_1_2_equal`, which should have an equal number of each target. Create a similar `X_1_2_equal`. The function `equalize_targets_binary` in the code chunk associated with Exercise 2.2.ii can be used. Remember to scale `X_1_2_equal`!  
#     iii. Do cross-validation with 5 stratified folds doing standard `LogisticRegression` (See Exercise 2.1.iv)  
#     iv. Do L2-regularisation with the following `Cs=  [1e5, 1e1, 1e-5]`. Use the same kind of cross-validation as in Exercise 2.2.iii. In the best-scoring of these models, how many more/fewer predictions are correct (on average)?  
#     v. Instead of fitting a model on all `n_sensors * n_samples` features, fit  a logistic regression (same kind as in Exercise 2.2.iv (use the `C` that resulted in the best prediction)) for __each__ time sample and use the same cross-validation as in Exercise 2.2.iii. What are the time points where classification is best? Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)  
#     vi. Now do the same, but with L1 regression - set `C=1e-1` - what are the time points when classification is best? (make a plot)?  
#     vii. Finally, fit the same models as in Exercise 2.2.vi but now for `data_1_4` and `y_1_4` (create a data set and a target vector that only contains PAS responses 1 and 4). What are the time points when classification is best? Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)

#%%%% Exercise 2.2.i

import sklearn
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

print(data_1_2.shape)
print(y_1_2.shape)

# Use the function
data_1_2_equal, y_1_2_equal = equalize_targets_binary(data_1_2, y_1_2)

print(data_1_2_equal.shape)
print(y_1_2_equal.shape)

# Reshape data into 2d
X_1_2_equal = data_1_2_equal.reshape(198, -1)

print(X_1_2_equal.shape)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
X_1_2_equal = scaler.fit_transform(X_1_2_equal)

#%%%% Exercise 2.2.iii

regressor = LogisticRegression(penalty = 'none')
log_fit_2 = regressor.fit(X_1_2_equal, y_1_2_equal)

skFold = sklearn.model_selection.cross_val_score(log_fit_2, X_1_2_equal, y_1_2_equal, cv=5)
print(skFold)

#accuracy score for each split 

#%%%% Exercise 2.2.iv

#same as above penalty = L2, logistic regression as a C- argument (opposite of lamda)
#do a for loop, for C in....
Cs=  [1e5, 1e1, 1e-5]
for c in Cs:
  log = LogisticRegression(penalty = 'l2', C=c) # solver default is lbfgs
  log_fit_equal = log.fit(X_1_2_equal, y_1_2_equal)
  scores = sklearn.model_selection.cross_val_score(log_fit_equal, X_1_2_equal, y_1_2_equal,cv=5)
  print(scores.mean())
  
  
#amount of predictions correct
from sklearn.model_selection import cross_val_predict as cvp
log_c_neg5 = LogisticRegression(penalty='l2', C=1e-5) 
predict_c_neg5 = cvp(log_c_neg5, X_1_2_equal, y_1_2_equal, cv=5)
accuracy_neg5 = predict_c_neg5 == y_1_2_equal  

##this is from Mina and I don't quite understand it, *** Write to Mina and ask :)
##accuracy Log Model 2.2iii
predict_log = cvp(regressor, X_1_2_equal, y_1_2_equal, cv=5)
accuracy_log = predict_log == y_1_2_equal
print("Correct Predictions Log 2.2iii:", len(np.where(accuracy_log == True)[0]))
print("Correct Predictions Log 1e-5:", len(np.where(accuracy_neg5 == True)[0]))

#based on the scores, Cs of 1e-5 is the most accurate, 60%, where the other two were 53% and 54%. We also have more correct predictions with the penalized model

#this way prints the whole array, not sure if that's needed but above code won't do it... Do we want the whole array?
Cs=  [1e5, 1e1, 1e-5]
for c in Cs:
  log = LogisticRegression(penalty = 'l2', C=c) # solver default is lbfgs
  log_fit_equal = log.fit(X_1_2_equal, y_1_2_equal)
  sklearn.model_selection.cross_val_score(log_fit_equal, X_1_2_equal, y_1_2_equal,cv=5)

#%%%% Exercise 2.2.v

## empty list for the cross scores 
cross_scores = []
for i in range(251):
  #creating data and scaling 
  scaler = StandardScaler()
  X_time = data_1_2_equal[:,:,i]
  X_time_scaled = scaler.fit_transform(X_time)
  
#creating a logistic regression object
  lr = LogisticRegression(penalty='l2', C=1e-5)
  
#cross-validating 
  score = sklearn.model_selection.cross_val_score(lr, X_time_scaled, y_1_2_equal, cv = 5)
  
#taking the mean 
  mean = np.mean(score)
  
#appending the mean
  cross_scores.append(mean)
  #print(cross_scores)
#on knit, don't include printed output

## FINDING the time point where classification is best ##
indexmax = cross_scores.index(max(cross_scores))
times[indexmax]
plt.figure()
plt.axvline(x = times[indexmax], color = "black", alpha = 0.5)  
plt.plot(times, cross_scores)
plt.axhline(y = 0.50, color = "black")
plt.title("L2 PAS 1 & 2: Classification Accuracy vs. Time")
plt.xlabel("Time (ms)")
plt.ylabel("Accuracy")
plt.show()

#the chance level is .5 or 50% because it is a binary classification, either it's pas 1/pas2 or not. 

#%%%% Exercise 2.2.vi

cross_scores_l1 = []
for i in range(251):
  #Creating data and scaling 
  scaler = StandardScaler()
  X_time = data_1_2_equal[:,:,i]
  X_time_scaled = scaler.fit_transform(X_time)
  logr = LogisticRegression(penalty='l1', solver = "liblinear", C=1e-1)
  score = sklearn.model_selection.cross_val_score(logr, X_time_scaled, y_1_2_equal, cv = 5)
  mean = np.mean(score)
  cross_scores_l1.append(mean)
  
indexmax_l1 = cross_scores_l1.index(max(cross_scores_l1))
times[indexmax_l1]
plt.figure()
plt.axvline(x = times[indexmax_l1], color = "black", alpha = 0.5)  
plt.plot(times, cross_scores_l1)
plt.axhline(y = 0.50, color = "black")
plt.title("L1 PAS 1 & 2: Classification Accuracy vs. Time")
plt.xlabel("Time (ms)")
plt.ylabel("Accuracy")
plt.show()

#%%%% Exercise 2.2.vii

pas14 = np.where((y == 1) | (y == 4))
data_1_4 = data[pas14] # np.squeeze gets rid of the point that is only 1
data_1_4.shape
data_1_4.ndim # how many dimensions 
y_1_4 = np.squeeze(y[pas14])
        
len(y_1_4)  
# equalize the data
data_1_4_equal, y_1_4_equal = equalize_targets_binary(data_1_4, y_1_4)
cross_scores_pas14 = []
for i in range(251):
  #Creating data and scaling 
  scaler = StandardScaler()
  X_time = data_1_4_equal[:,:,i]
  X_time_scaled = scaler.fit_transform(X_time)
  logr = LogisticRegression(penalty='l1', solver = "liblinear", C=1e-1)
  score = sklearn.model_selection.cross_val_score(logr, X_time_scaled, y_1_4_equal, cv = 5)
  mean = np.mean(score)
  cross_scores_pas14.append(mean)
  
indexmax_pas14 = cross_scores_pas14.index(max(cross_scores_pas14))
times[indexmax_pas14]
plt.figure()
plt.axvline(x = times[indexmax_l1], color = "black", alpha = 0.5)  
plt.plot(times, cross_scores_pas14)
plt.axhline(y = 0.50, color = "black")
plt.title("L1 PAS 1 & 4: Classification Accuracy vs. Time")
plt.xlabel("Time (ms)")
plt.ylabel("Accuracy")
plt.show()

#%%% Exercise 2.3
# 3) Is pairwise classification of subjective experience possible? Any surprises in the classification accuracies, i.e. how does the classification score fore PAS 1 vs 4 compare to the classification score for PAS 1 vs 2?

#we expect to see more of a difference between pas 1 and 4 than pas 1 and 2 but we don't

#%% EXERCISE 3 - Do a Support Vector Machine Classification on all four PAS-ratings

#%%% Exercise 3.1
# 1) Do a Support Vector Machine Classification  
#     i. First equalize the number of targets using the function associated with each PAS-rating using the function associated with Exercise 3.1.i  
#     ii. Run two classifiers, one with a linear kernel and one with a radial basis (other options should be left at their defaults) - the number of features is the number of sensors multiplied the number of samples. Which one is better predicting the category?
#     iii. Run the sample-by-sample analysis (similar to Exercise 2.2.v) with the best kernel (from Exercise 3.1.ii). Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)
#     iv. Is classification of subjective experience possible at around 200-250 ms?

#%%%% Exercise 3.1.i

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

#making a dataframe
data_equal, y_equal = equalize_targets(data, y)
print(data_equal.shape)
print(y_equal.shape)

#transform data to 2d data
data_equal_2d = data_equal.reshape(data_equal.shape[0],-1) 

#scaling it
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data_equal_scale = sc.fit_transform(data_equal_2d)
print(data_equal_scale)
data_equal_scale.shape

#%%%% Exercise 3.1.ii

from sklearn.svm import SVC

svm_linear = SVC(kernel ='linear')
svm_radial = SVC(kernel ='rbf')

# cross validating the linear support vector
svm_linear_scores = sklearn.model_selection.cross_val_score(svm_linear, data_equal_scale, y_equal, cv=5)
# cross validating the radial support vector
svm_radial_scores = sklearn.model_selection.cross_val_score(svm_radial, data_equal_scale, y_equal, cv=5)
## printing the mean of the cross-validated performances
print("SVM Linear Mean Cross Validated:", round(np.mean(svm_linear_scores), 3))
print("SVM Radial Mean Cross Validated:", round(np.mean(svm_radial_scores), 3))

#the numbers aren't great, but the radial looks a bit better
#%%%% Exercise 3.1.iii

## empty list for the cross scores 
cross_scores2 = []
cv = StratifiedKFold(n_splits=5)
for i in range(251):
  sc = StandardScaler()
  X_time2 = data_equal[:,:,i]
  X_time2_scale = sc.fit_transform(X_time2)
  
#cross-validating 
  score = sklearn.model_selection.cross_val_score(svm_radial, X_time2_scale, y_equal, cv = 5)
  
#taking the mean 
  mean = np.mean(score)
  
#appending the mean
  cross_scores2.append(mean)
  
#looking for the module where the best classification is
indexmax = cross_scores2.index(max(cross_scores2))
times[indexmax]

plt.figure()
plt.axvline(x = times[indexmax], color = "black", alpha = 0.7)
plt.axhline(y = 0.25, color = "black")
plt.plot(times, cross_scores2)
plt.title("SVC with Radial Basis on all PAS: Classification Accuracy vs. Time")
plt.xlabel("Time (ms)")
plt.ylabel("Accuracy")
plt.show()

print(times[indexmax])

#%%%% Exercise 3.1.iv

#yes because it's above the chance level (see plot)

#%%% Exercise 3.2
# 2) Finally, split the equalized data set (with all four ratings) into a training part and test part, where the test part if 30 % of the trials. Use `train_test_split` from `sklearn.model_selection`  
#     i. Use the kernel that resulted in the best classification in Exercise 3.1.ii and `fit`the training set and `predict` on the test set. This time your features are the number of sensors multiplied by the number of samples.  
#     ii. Create a _confusion matrix_. It is a 4x4 matrix. The row names and the column names are the PAS-scores. There will thus be 16 entries. The PAS1xPAS1 entry will be the number of actual PAS1, $y_{pas1}$ that were predicted as PAS1, $\hat y_{pas1}$. The PAS1xPAS2 entry will be the number of actual PAS1, $y_{pas1}$ that were predicted as PAS2, $\hat y_{pas2}$ and so on for the remaining 14 entries.  Plot the matrix
#     iii. Based on the confusion matrix, describe how ratings are misclassified and if that makes sense given that ratings should measure the strength/quality of the subjective experience. Is the classifier biased towards specific ratings?

#importing package
import sklearn.model_selection as sklearn

#splitting into training and test parts
x_train, x_test, y_train, y_test = sklearn.train_test_split(data_equal_2d, y_equal, test_size=0.3, random_state = 12)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#%%%% Exercise 3.2.i

svm_radial.fit(x_train, y_train)
predicted_y = svm_radial.predict(x_test)
print(predicted_y)

#%%%% Exercise 3.2.ii

from sklearn.metrics import ConfusionMatrixDisplay 
ConfusionMatrixDisplay.from_predictions(y_test, predicted_y)
plt.show()

#%%%% Exercise 3.2.iii

#For PAS1-ratings half of the predtictions were correctly classified. It makes good sense that most of the predictions for PAS1 lies within PAS 1 and 2, as PAS 3 and 4 means a very clear experience for the participants.
#The classifier is defnitely biased towards PAS 1 and 2.
#Overall, the classifier does not perform very well. This, though, makes good sense as the radial bases kernel had an accuracy score of only 33%. Both the radial bases and the linear kernel classifiers had poor accuracy rates — but the radial bases kernel performed best and thus, we chose to use this one.