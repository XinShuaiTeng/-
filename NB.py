# 基于贝叶斯的邮件分类系统
import jieba
import re
import pandas as pd
import math
from collections import Counter

#标准化数据
def normalizing(text):
	email_pattern=re.compile('[0-9A-Za-z_-]*@[0-9a-zA-Z]+\.[A-Za-z.]*')
	text=re.sub(email_pattern,"邮箱号",text)
	qq_pattern=re.compile('[1-9][0-9]{7,12}')
	text=re.sub(qq_pattern,"扣扣号",text)
	http_pattern=re.compile('(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?')
	text=re.sub(http_pattern,"网址",text)
	else_pattern=re.compile('[0-9a-zA-Z+_#，。 ,./`;*\-+!@#$%^&*()_+":：“】\n【].*?')
	text=re.sub(else_pattern," ",text)
	return text
#除去停用词
def remove_stopwords(stop_word,dic):
	stop_word.append('\n')
	stop_word.append(' ')
	for x in stop_word:
		if  dic.__contains__(x) :
			# print(x)
			del dic[x]
	return dic

#数据预处理
def data_preprocessing(_text):
	_text=normalizing(_text)#标准化手机号 网址 邮箱 等数据
	_ls=jieba.cut(_text)
	_dic=Counter(list(_ls))
	_dic=remove_stopwords(stop_word,_dic)#除去停用词
	#转为dataframe 然后排序，有更好的方法可代替此处
	_df=pd.DataFrame(list(_dic.values()),index=_dic.keys())
	_df=_df.sort_index(ascending=False)
	#去除低频词，取高频前5000
	return _df[:5000]
#读取文件
stop_word=[]
normal_email=[]
abnormal_email=[]
email_test=[]
with open('corpurs/ham_5000.utf8','r',encoding="utf-8") as f:
	for x in f:
		normal_email.append(x.rstrip())
with open('corpurs/spam_5000.utf8','r',encoding="utf-8") as f:
	for x in f:
		abnormal_email.append(x.rstrip())
with open('corpurs/stop_word.txt','r',encoding="utf-8") as f:
	for x in f:
		stop_word.append(x.rstrip())
with open('corpurs/email_test.utf8','r',encoding="utf-8") as f:
	for x in f:
		email_test.append(x.rstrip())

N=len(normal_email)
M=len(abnormal_email)

p_abnormal=N/(M+N)
p_no=1-p_abnormal

abnormal_email_text=" ".join(abnormal_email)
normal_email_text=" ".join(normal_email)

#数据预处理

abnormal_email_dict=data_preprocessing(abnormal_email_text).to_dict()[0]
normal_email_dict=data_preprocessing(normal_email_text).to_dict()[0]

total=set(abnormal_email_dict.keys()) | set(normal_email_dict.keys())

'''
 重新计算模型

total_dic={ '单词'：[广告邮件出现次数,正常邮件出现次数]  }

'''
total_dic={}
for x in list(total):
	temp=[0,0]
	if abnormal_email_dict.__contains__(x):
		temp[0]=abnormal_email_dict[x]
	if normal_email_dict.__contains__(x):
		temp[1]=normal_email_dict[x]
	total_dic[x]=temp

# print(total_dic)




## 重写log函数
def log(p):
	if p<=0:#处理非法数据
		p=0.000001
	if p==1.0:#处理概率为1 的数据
		p=p-0.000001
	return math.log(p)

##计算条件概率
def condition_probability(word,y=1):
	#正常 y=1 广告 y=0
	if total_dic.__contains__(word):
		temp=total_dic[word]
		# print(word,temp)
		p=(temp[y]+1)/(2*(temp[0]+temp[1]))
		# print(word,p)
		return p
	# 如果字典中不存在，使用先验概率代替
	return p_no if y==1 else p_abnormal

#贝叶斯建模
def bayes(content):
	content=normalizing(content)
	words=jieba.cut(content)
	correct,incorrect=log(p_no),log(p_abnormal)
	for w in words:
		if w=="" or w==" ":
			continue
		#正常 y=1 广告 y=0
		correct+=log(condition_probability(w,y=1))
		incorrect+=log(condition_probability(w,y=0))
	# print(correct,incorrect)
	return 1 if correct>incorrect else 0


#评估系统

res=[]
fact=[0 for _ in range(50)]+[1 for _ in range(50)]

for  content in email_test:
	res.append(bayes(content))
print("\n==================测试结束===================\n")
print(f"测试数目：{len(res)}")

TP,FP,TN,FN=0,0,0,0

for i in range(len(fact)):
	if fact[i]==res[i]:
		if fact[i]==1:
			TP+=1
		else:
			TN+=1
	else:
		if fact[i]==1 :
			FN+=1
		else:
			FP+=1
precision=100*TP/(TP+FP)
recall=100*TP/(TP+FN)
f1=(2*precision*recall)/(precision+recall)
print(f"正确率：{TP+TN}%   \n精确率：{precision}% \n召回率：{recall}%\nF1-measure：{f1}%")
print("\n==================测试结束===================\n")
