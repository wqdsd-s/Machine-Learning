import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


data=pd.read_csv('C:\\Users\\Downloads\\imports-85.csv')
data['log_price']=np.log(data['price'])
data['squared_price']=data['price']**2
print(data)

data=data.merge(pd.get_dummies(data['body-style']),how='left',left_index=True,right_index=True)


for bodystyle,color in [('convertible','red'),('hardtop','blue'),('hatchback','yellow'),('sedan','black'),('wagon','green')]:
    temp_data=data[data['body-style']==bodystyle]
    plt.scatter(temp_data['horsepower'],temp_data['price'],c=color,label=bodystyle,linewidths=.05)
    
plt.xlabel("horsepower")
plt.ylabel("price")
plt.legend()
plt.savefig(r'C:\\Users\\alex\\Desktop\\UCLAmachine\\1.png')
plt.show()

for bodystyle,color in [('convertible','red'),('hardtop','blue'),('hatchback','yellow'),('sedan','black'),('wagon','green')]:
    temp_data=data[data['body-style']==bodystyle]
    plt.scatter(temp_data['horsepower'],temp_data['log_price'],c=color,label=bodystyle,linewidths=.05)
    
plt.xlabel("horsepower")
plt.ylabel("log_price")
plt.legend()
plt.show()

for bodystyle,color in [('convertible','red'),('hardtop','blue'),('hatchback','yellow'),('sedan','black'),('wagon','green')]:
    temp_data=data[data['body-style']==bodystyle]
    plt.scatter(temp_data['horsepower'],temp_data['squared_price'],c=color,label=bodystyle,linewidths=.05)
    
plt.xlabel("horsepower")
plt.ylabel("squared_price")
plt.legend()
plt.show()


mod = sm.OLS(data['log_price'],data[['horsepower','convertible','hardtop','hatchback','sedan','wagon']],missing='drop')
res=mod.fit()
print(res.summary())

mod1 = sm.OLS(data['squared_price'],data[['horsepower','convertible','hardtop','hatchback','sedan','wagon']],missing='drop')
res1=mod1.fit()
print(res1.summary())


data['cons']=1
mod2 = sm.OLS(data['log_price'],data[['horsepower','cons']],missing='drop')
res2=mod2.fit()
print(res2.summary())

plt.scatter(data['horsepower'],data['log_price'],linewidths=.05)
plt.plot(data['horsepower'][res2.fittedvalues.index],res2.fittedvalues,label='fitted value')
plt.xlabel("horsepower")
plt.ylabel("log_price")
plt.legend()
plt.show()

'''
from seaborn_qqplot import pplot
pplot(data, x="horsepower", y="log_price",kind='qq',display_kws={"identity":False, "fit":True})

'''

plt.scatter(data['horsepower'],data['city-mpg'],linewidths=.05)
plt.xlabel("horsepower")
plt.ylabel("city-mpg")
plt.show()
mod3 = sm.OLS(data['city-mpg'],data[['horsepower','cons']],missing='drop')
res3=mod3.fit()
print(res3.summary())

