# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 19:54:30 2018

@author: abc
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
data=pd.read_csv('XYZCorp_LendingData.txt',sep='\t',low_memory = False)
#%%
df=pd.DataFrame.copy(data)
#%%
#data_types=df.dtypes.value_counts()
#data_types
#%%
df_missing=df.isnull().sum().reset_index()
df_missing.columns=['Col_Name','Num_of_MV']
df_missing=df_missing[df_missing['Num_of_MV']>0]
df_missing=df_missing.sort_values(by='Num_of_MV',ascending=False)
df_missing['Percentage']=(df_missing['Num_of_MV']/len(df))*100
df_missing=df_missing.reset_index()   #
Max_Missing=df_missing.iloc[0:21,1].values   ##iloc use for dropping columns with greater than 50% missing values
df=df.drop(Max_Missing,axis=1)
#%%
df= df[~((df['tot_coll_amt'].isnull()) & (df['default_ind'] == 0)) ]
#%%
df.isnull().sum()
#%%
df['tot_coll_amt']=df['tot_coll_amt'].fillna(np.mean(df['tot_coll_amt']))
df['tot_cur_bal']=df['tot_cur_bal'].fillna(np.mean(df['tot_cur_bal']))
df['total_rev_hi_lim']=df['total_rev_hi_lim'].fillna(np.mean(df['total_rev_hi_lim']))
df['collections_12_mths_ex_med']=df['collections_12_mths_ex_med'].fillna(0)
df['revol_util']=df['revol_util'].fillna(np.mean(df['revol_util']))
#%%
df=df.drop(df[["next_pymnt_d","title","emp_title"]],axis=1)
#%%
df.emp_length.replace(['< 1 year'],'0 years',inplace=True)
df['emp_length'].fillna("0 years",inplace = True)
df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)
#%%
df['term']=df['term'].apply(lambda x:x.split()[0])
df['term']=pd.to_numeric(df['term'])
df["term"].head()
#%%
df['initial_list_status']=df['initial_list_status'].apply(lambda x: 1 if "f" in x else 0)
df['application_type']=df['application_type'].apply(lambda x: 1 if "INDIVIDUAL" in x else 0)
df['pymnt_plan']=df['pymnt_plan'].apply(lambda x: 1 if "n" in x else 0)
#%%
df= df[~((df['last_pymnt_d'].isnull()) & (df['default_ind'] == 0)) ]
df= df[~((df['last_credit_pull_d'].isnull()) & (df['default_ind'] == 0)) ]
#%%
df['last_pymnt_d']=df['last_pymnt_d'].fillna(df['last_pymnt_d'].mode()[0])
df['last_credit_pull_d']=df['last_credit_pull_d'].fillna(df['last_credit_pull_d'].mode()[0])
#%%
df.isnull().sum()
#%%
corr_df=df.corr(method='pearson')
print(corr_df)
sns.heatmap(corr_df,vmax=1.0,vmin=-1.0)
#%%
plt.figure(figsize=(24,9))
sns.heatmap(df.corr(), annot=True, linewidths=.8,fmt='.2f')
plt.title('Correlation Heat Map')
#%%
df=df.drop(['id','member_id','policy_code','loan_amnt','funded_amnt','installment','total_rec_prncp',
            'out_prncp', 'total_pymnt',"sub_grade","recoveries"],axis=1)
#%%
df['default_ind'].hist(bins=8)
# from histogram non defaulters are more than the number of defaulters
#bins changes the width of the bars. higher the bin lower the width
#%%

df.last_pymnt_d =pd.to_datetime(df.last_pymnt_d)
df.last_credit_pull_d =pd.to_datetime(df.last_credit_pull_d)
df['earliest_cr_line']=pd.to_datetime(df['earliest_cr_line'])
df['issue_d']=pd.to_datetime(df['issue_d'])
#%%
current_time=df.last_pymnt_d.max()
current_time
df['Time_diff_def']=df.last_pymnt_d.map(lambda x: current_time-x) 

df['Time_diff_days_def']=df.Time_diff_def.map(lambda x: x.days)
#%%
#2. Creating number of days out of existing credit line date
current_time=df.earliest_cr_line.max()
current_time
df['Time_diff']=df.earliest_cr_line.map(lambda x: current_time-x) 
df['Time_diff_days_cr']=df.Time_diff.map(lambda x: x.days)
#%%
current_time=df.last_credit_pull_d.max()
current_time
df['Time_diff_pull']=df.last_credit_pull_d.map(lambda x: current_time-x) 
df['Time_diff_days_pull']=df.Time_diff_pull.map(lambda x: x.days)


#%%
df=df.drop(["zip_code","Time_diff","Time_diff_pull","Time_diff_def","last_credit_pull_d",
            "earliest_cr_line","last_pymnt_d"],axis =1)
#%%
#df['issue_d']=pd.to_datetime(df['issue_d'])
train_data=df[df['issue_d']<'2015-06-01']
test_data=df[df['issue_d']>='2015-06-01']
#%%
train_data=train_data.drop(["issue_d"],axis =1)
test_data=test_data.drop(["issue_d"],axis =1)
#%%
#Upsampling of Training Data
from sklearn.utils import resample
#%%
# Separate majority and minority classes
train_majority = train_data[train_data.emp_length==0]
train_minority = train_data[train_data.emp_length==1]
#%%
# Upsample minority class
train_minority_upsampled = resample(train_minority,replace=True,
                                    n_samples=552529,
                                    random_state=123) # reproducible results

#%%
# Combine majority class with upsampled minority class
train_upsampled = pd.concat([train_majority, train_minority_upsampled])
#%%
# Display new class counts
train_upsampled.default_ind.value_counts()

#%%
colname = ["addr_state","grade",
           "home_ownership","purpose","verification_status"]
from sklearn import preprocessing

le ={} #creating an emplty dictionary{f:0,M:1}.key value pairs

for x in colname:
    le[x] = preprocessing.LabelEncoder()  #labelencoder generates labels 


for x in colname:
    #data[x] = data[x].astype(str)
    train_data[x] = le[x].fit_transform(train_data.__getattr__(x))  #fit_transform is asscoiated with labelencode method


#fetching the labels and replacing the values in dataframes with those labels
for x in colname:
    #data[x] = data[x].astype(str)
    test_data[x] = le[x].fit_transform(test_data.__getattr__(x))  #fit_transform is asscoiated with labelencode method
train_data.head()
#%%
X=train_data.drop('default_ind',axis=1)
y=train_data['default_ind']
test_X=test_data.drop('default_ind',axis=1)
test_y=test_data['default_ind']
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  #standard scaler will give values which is held in scaler

scaler.fit(X)  #find out mapped values
scaler.fit(test_X)
X=scaler.transform(X)  #replaces the value with mapped values
test_X=scaler.transform(test_X)
print(X)
#%%
y=y.astype(int)  #as sometimes it creates an object and when you feed an obejct to a model it gives an errory
test_y = test_y.astype(int)
#%%
from sklearn.linear_model import LogisticRegression
#create a model
classifier = (LogisticRegression())  #object classifier
#fitting training data to the model
classifier.fit(X,y)  #fit trains the model. 1st has to be x and then y. for statslearn instead of sklear 1st y then x
#for the x values these are the y values

y_pred = classifier.predict(test_X)  
#print(list(zip(test_y,y_pred)))  #zip is to merge 2 list
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm = confusion_matrix(test_y,y_pred)
print(cfm, "\n")

print("Classification report : ")

print(classification_report(test_y,y_pred))

acc=accuracy_score(test_y,y_pred)
print("accuracy of the model : ",acc)
#%%
y_pred_prob = classifier.predict_proba(test_X)
print(y_pred_prob)
#%%
for a in np.arange(0,1,0.02):
    predict_mine = np.where(y_pred_prob[:,0]<a,1,0)
    cfm = confusion_matrix(test_y.tolist(),predict_mine)
    total_err = cfm[0,1]+cfm[1,0]
    print("Errors at threshold : ",a ,":", total_err, "type 2 error : ", cfm[1,0] , "type 1 error : ",cfm[0,1])
#%%
y_pred_class = []
for value in y_pred_prob[:,0] :  #using index 0 values only
    if value <0.98:
        y_pred_class.append(1)
    else :
        y_pred_class.append(0)


#y_pred_class
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm = confusion_matrix(test_y.tolist(),y_pred_class)
print(cfm, "\n")

acc=accuracy_score(test_y.tolist(),y_pred_class)
print("accuracy of the model : ",acc)

print("Classification report : ")

print(classification_report(test_y.tolist(),y_pred_class))

#%%
#Random forest
from sklearn.ensemble import RandomForestClassifier

model_RandomForest = RandomForestClassifier(25,random_state = 10) #no 479 is the no of trees
#fit the model on the data and predict

model_RandomForest.fit(X,y)

y_pred = model_RandomForest.predict(test_X)

#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm = confusion_matrix(test_y,y_pred)
print(cfm, "\n")

print("Classification report : ")

print(classification_report(test_y,y_pred))

acc=accuracy_score(test_y,y_pred)
print("accuracy of the model : ",acc)
#%%
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(test_y, y_pred)
auc = metrics.auc(fpr,tpr)
print(auc)
#%%
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




