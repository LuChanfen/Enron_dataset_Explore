#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','exercised_stock_options','total_stock_value','salary','bonus','deferred_income'] 
# You will need to use more features    
# all_features_list contains all features in the dataset
all_features_list = ['poi','salary','bonus','deferral_payments','total_payments','deferred_income','total_stock_value','expenses','exercised_stock_options','other','long_term_incentive','restricted_stock','to_messages','from_poi_to_this_person','from_messages','from_this_person_to_poi','shared_receipt_with_poi']

# new_features_list contains features in features_list and the new features that I create 
new_features_list = ['poi','salary','bonus','exercised_stock_options','ratio_from_poi','ratio_to_poi','total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
       
        
### Store to my_dataset for easy export below.
my_dataset = data_dict

# calclate number of point in dmy_dataset
data_length = 0
features_num = 0
for i in my_dataset.keys():    
    for j in range(len(my_dataset[i])):
        data_length = data_length + 1
print "------------------number of dataset------------------------------------------------------------"
print "the number of dataset: ", data_length
print "the number of the person:", len(my_dataset) 

print "------------------number of NaN detection-------------------------------------------------------"
number_of_poi = 0
number_of_notpoi = 0
number_of_neither = 0
number_of_nan = dict()
nan_salary_poi = 0
nan_total_payments_poi = 0
nan_email_address_poi = 0
nan_bonus_poi = 0
for i in my_dataset[my_dataset.keys()[0]].keys():
    if i == 'poi':
        continue
    else:
        number_of_nan[i] = 0
        
for i in my_dataset.keys():
    if my_dataset[i]['poi'] == 1:
        number_of_poi = number_of_poi +1
        if my_dataset[i]['salary'] == 'NaN':
            nan_salary_poi = nan_salary_poi+1
        if my_dataset[i]['total_payments'] == 'NaN':
            nan_total_payments_poi = nan_total_payments_poi+1
        if my_dataset[i]['email_address'] == 'NaN':
            nan_email_address_poi = nan_email_address_poi+1
        if my_dataset[i]['bonus'] == 'NaN':
            nan_bonus_poi = nan_bonus_poi+1
    elif my_dataset[i]['poi'] == 0:
        number_of_notpoi = number_of_notpoi + 1
    else:
        number_of_neither = number_of_neither +1
    for j in my_dataset[i].keys():
        if my_dataset[i][j] == "NaN" and j != "poi":
            number_of_nan[j] = number_of_nan[j] + 1
features_num = 0
auct_features_list = []
for i in my_dataset.keys():
    for index,item in enumerate(my_dataset[i]):
        if item == "poi":
            continue
        else:
            auct_features_list.append(item)
            features_num = features_num +1
    break
print "number of poi: ", number_of_poi
print "number of not poi: ", number_of_notpoi
print "number of neither poi or not poi: ", number_of_neither
print "number of features: ", features_num
print "the list of features:", auct_features_list
print "-----------------------------------------------------------------------------------------------"

print "number of person's salary is 'NaN':",number_of_nan['salary'],",and number of poi's salary is 'NaN':", nan_salary_poi,"the ratio is:",round(float(nan_salary_poi)/len(my_dataset)*100,3),"%"
print "number of person's total_payments is 'NaN':",number_of_nan['total_payments'],",and number of poi's total_payments is 'NaN':", nan_total_payments_poi,"the ratio is :",round(float(nan_total_payments_poi)/len(my_dataset)*100,3),"%"
print "number of person's email_address is 'NaN':",number_of_nan['email_address'],",and number of poi's email_address is 'NaN':", nan_email_address_poi,"the ratio is:",round(float(nan_email_address_poi)/len(my_dataset)*100,3),"%"
print "number of person's bonus is 'NaN':",number_of_nan['bonus'],",and number of poi's bonus is 'NaN':", nan_bonus_poi,"the ratio is:",round(float(nan_bonus_poi)/len(my_dataset)*100,3),"%"
print "-----------------------------------------------------------------------------------------------"

number_of_nan = sorted(number_of_nan.items(), lambda x,y:cmp(x[1],y[1]), reverse = True)    
print "features that have 'NaN' and its number:"
print number_of_nan

# remove thrid high NaN features for each person
for i in my_dataset.keys():
    for j in my_dataset[i].keys():
        if j in [number_of_nan[0][0], number_of_nan[1][0], number_of_nan[2][0]]:
            my_dataset[i].pop(j)

print "-----------------------------------------------------------------------------------------------"


### Task 2: Remove outliers and point distribution
print "-----------------","outlier","-----------------------------------------------------------------"

# use visualization to detect outlier and 
def draw_point(data, label1_name,label1_num, label2_name, label2_num,title):
    poi_data = []
    non_poi_data = []
    for i in range(len(data)):
        if data[i][0] == 0:
            non_poi_data.append(data[i])
        elif data[i][0] == 1:
            poi_data.append(data[i])
    for point in non_poi_data[1:]:
        a = point[label1_num]
        b = point[label2_num]
        plt.scatter(a,b,color='b')
    for point in poi_data[1:]:
        a = point[label1_num]
        b = point[label2_num]
        plt.scatter(a,b,color='r')
    plt.scatter(non_poi_data[0][label1_num], non_poi_data[0][label2_num],color='b',label='non_poi')
    plt.scatter(poi_data[0][label1_num], poi_data[0][label2_num], color='r', label='poi')
    plt.xlabel(label1_name)
    plt.ylabel(label2_name)
    plt.title(title)
    plt.legend()
    plt.show()

# explore whether some person's features are all NaN
nan_person = []
num_nan = 0
for i in my_dataset.keys():
    for j in my_dataset[i].keys():
        if j == "poi":
            continue
        elif my_dataset[i][j] == "NaN":
            num_nan = num_nan + 1
        if num_nan == (len(my_dataset[i])-1):
            nan_person.append(i)
    num_nan = 0
print "the person  whose featuers are all  NaN:",nan_person
        

# Extract features and labels from dataset for local testing  
data = featureFormat(my_dataset, all_features_list, sort_keys = True)
draw_point(data, "salary",1,"bonus",2,"salary & bonus with outlier")


# remove the outlier and extract features and labels again
my_dataset.pop('TOTAL',0)
my_dataset.pop("TRAVEL AGENCY IN THE PARK",0)
my_dataset.pop(nan_person[0],0)
data = featureFormat(my_dataset, all_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

draw_point(data, "salary",1, "bonus",2,"salary & bonus without outlier")


print "-----------------------------------------------------------------------------------------------"


### Task 3: Create new feature(s)
print "---------------------new features--------------------------------------------------------------"

ratio_from_poi = []
ratio_to_poi = []
for i in my_dataset.keys():
    if my_dataset[i]['from_poi_to_this_person'] == 'NaN' or my_dataset[i]['to_messages'] == 'NaN':
        ratio_from_poi.append(0.0)
    else:
        ratio_from_poi.append( float(my_dataset[i]['from_poi_to_this_person'])/my_dataset[i]['to_messages'])
    if my_dataset[i]['from_this_person_to_poi'] == 'NaN' or my_dataset[i]['from_messages'] == 'NaN':
        ratio_to_poi.append(0.0)
    else:
        ratio_to_poi.append( float(my_dataset[i]['from_this_person_to_poi'])/my_dataset[i]['from_messages'])

num = 0
# create new dataset which contains new features and new_dataset can help to test 
# whether new features are good to final algorithm
new_dataset = my_dataset.copy()
for i in my_dataset.keys():
    new_dataset[i]['ratio_from_poi'] = ratio_from_poi[num]
    new_dataset[i]['ratio_to_poi'] = ratio_to_poi[num]
    num = num + 1
    
new_data = featureFormat(new_dataset, new_features_list, sort_keys = True)
new_labels, new_features = targetFeatureSplit(new_data)

draw_point(new_data, "ratio_from_poi", 4, "ratio_to_poi", 5,"ratio_from_poi & ratio_to_poi")

print "-----------------------------------------------------------------------------------------------"

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

# use SelectKBest to select best features
print "--------------------SelectKBest features scores------------------------------------------------"

all_data = featureFormat(my_dataset, all_features_list, sort_keys = True)
all_labels, all_features = targetFeatureSplit(all_data)
select_features = SelectKBest(f_classif, k=5).fit(all_features, all_labels)
faetures = select_features.transform(all_features)

features_dict = dict()
for i in range(1,len(all_features_list)):
    features_dict[all_features_list[i]] = round(select_features.scores_[i-1],3)

features_dict = sorted(features_dict.items(), lambda x,y:cmp(x[1],y[1]),reverse=True)
for i in range(len(features_dict)):
    print features_dict[i]
print "-----------------------------------------------------------------------------------------------"

#print "------------------------split train test-------------------------------------------------------" 

# split my_dataset into training set and testing set
features_train = []
features_test = []
labels_train = []
labels_test = []
sss = StratifiedShuffleSplit(labels, n_iter = 100,test_size=0.3, random_state=42)
for train_index, test_index in sss:
    for item in train_index:
        features_train.append(features[item])
        labels_train.append(labels[item])
    for item in test_index:
        features_test.append(features[item])
        labels_test.append(labels[item])


# split new_dataset into training set and testing set
new_features_train = []
new_features_test = []
new_labels_train = []
new_labels_test = []
new_sss = StratifiedShuffleSplit(new_labels, n_iter = 100, test_size=0.3, random_state=42)
for train_index, test_index in new_sss:
    for item in train_index:
        new_features_train.append(new_features[item])
        new_labels_train.append(new_labels[item])
    for item in test_index:
        new_features_test.append(new_features[item])
        new_labels_test.append(new_labels[item])

#print "------------------------features scalers-------------------------------------------------------" 
 
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
#scaler = MinMaxScaler()
#scaler = StandardScaler()
#features_train = scaler.fit_transform(features_train)
#features_test = scaler.transform(features_test)

    
    
print "------------------------classifiers------------------------------------------------------------" 



# use SelectKest to select the best 2 features
#select = SelectKBest(k=2)
#new_select = SelectKBest(k=2)

# use GridSearchCV 
#param_grid = {'min_samples_split':[2,5,10],
#              'min_samples_leaf':[1,2,10]}
param_grid = {}
clf = GridSearchCV(GaussianNB(),param_grid = param_grid)
clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)

new_clf = GridSearchCV(GaussianNB(), param_grid=param_grid)
new_clf.fit(new_features_train, new_labels_train)
new_labels_pred = new_clf.predict(new_features_test)

print "the best estimator:",clf.best_estimator_
print "the best score:",round(clf.best_score_,3)
print "---------------------------------------"
print "new features with best estimator:",new_clf.best_estimator_
print "new features with best score:",round(new_clf.best_score_,3)


print "detailed classification report: "
print classification_report(labels_test, labels_pred)
print "-----------------------------------------------------------------------------------------------"
print "detailed classification report with new features:"
print classification_report(new_labels_test, new_labels_pred)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)




