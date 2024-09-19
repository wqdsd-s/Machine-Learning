import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

data=pd.read_csv('C:\\Users\\alex\\Downloads\\LendingClub_LoanStats3a_v12.csv')
print(data)
data=data[data["loan_status"].apply(lambda x:x=='Fully Paid' or x=='Charged Off')]
data['Default']=data["loan_status"].apply(lambda x:1 if x=='Charged Off' else 0)
def grade_to_float(x):
    if x=='A':
        return 7
    elif x=='B':
        return 6
    elif x=='C':
        return 5
    elif x=='D':
        return 4
    elif x=='E':
        return 3
    elif x=='F':
        return 2
    elif x=='G':
        return 1
    
data['grade']=data["grade"].apply(grade_to_float)

data['cons']=1
model=sm.Logit(data['Default'],data[["grade",'cons']]).fit()
print(model.summary())

model1=sm.Logit(data['Default'],data[['cons']]).fit()
print(model1.summary())

print(-2*(model1.llf-model.llf))
print(1-chi2.cdf(-2*(model1.llf-model.llf), 1))


lgt_fpr, lgt_tpr, _=roc_curve(data['Default'],model.predict(data[["grade",'cons']]))
random_fpr, random_tpr, _ = roc_curve(data['Default'], [0 for i in range(len(data['Default']))])

plt.plot(random_fpr, random_tpr, linestyle='--', label='No Skill')
plt.plot(lgt_fpr, lgt_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()

          

data['prob']=model.predict(data[["grade",'cons']])
cutoff_list=[0]+sorted(model.predict(data[["grade",'cons']]).unique())
profit_list=[]
for p in cutoff_list:
    data['predict_default']=data['prob'].apply(lambda x:0 if x<=p else 1)
    temp_data=data[data['predict_default']==0]
    profit_list.append(np.sum(temp_data['Default'].apply(lambda x:1 if x==0 else -10)))

print(cutoff_list)
print(profit_list)


plt.annotate('maximum profit',(lgt_fpr[-2],lgt_tpr[-2]),xytext =(lgt_fpr[-2]-0.4,lgt_tpr[-2]),arrowprops=dict(arrowstyle='->'))

# show the plot
plt.show()
