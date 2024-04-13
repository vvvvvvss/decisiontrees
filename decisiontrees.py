import pandas as pd
df = pd.read_csv("ds_salaries.csv")
df.head()
inputs = df.drop('salary',axis='columns')
target = df['salary']
from sklearn.preprocessing import LabelEncoder
work = LabelEncoder()
experiencelevel = LabelEncoder()
employmenttype = LabelEncoder()
jobtitle = LabelEncoder()
remoteratio = LabelEncoder()
companyloc = LabelEncoder()
companysize = LabelEncoder()
inputs['work'] = work.fit_transform(inputs['work_year'])
inputs['experiencelevel'] = experiencelevel.fit_transform(inputs['experience_level'])
inputs['employmenttype'] = employmenttype.fit_transform(inputs['employment_type'])
inputs['jobtitle'] = jobtitle.fit_transform(inputs['job_title'])
inputs['remoteratio'] = remoteratio.fit_transform(inputs['remote_ratio'])
inputs['companyloc'] = companyloc.fit_transform(inputs['company_location'])
inputs['companysize'] = companysize.fit_transform(inputs['company_size'])
inputs
inputs_n = inputs.drop(['work_year','experience_level','employment_type','job_title','remote_ratio','company_location','company_size'],axis='columns')
inputs_n
target
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n,target)
model.score(inputs_n,target)
