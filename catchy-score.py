import pickle
from nltk.corpus import stopwords 

with open("C:/Users/Dell_Owner/Desktop/ytokenizer.pickle",'rb') as handle:
	ytokenizer = pickle.load(handle)

with open("C:/Users/Dell_Owner/Desktop/xtokenizer.pickle",'rb') as handle:
	xtokenizer = pickle.load(handle)

dict1 = ytokenizer.word_counts
dict2 = xtokenizer.word_counts

def merge_two_dicts(x, y):
    z = x.copy()   
    z.update(y)    
    return z

dict_new = merge_two_dicts(dict1,dict2)

def catchyscore(ground,catchy):
	ground_new = []
	catchy_new = []
	for i in range (len(ground)):
		if ground[i] not in stopwords:
			ground_new.append(ground[i])
		else:
			continue
	for i in range (len(catchy)):
		if catchy[i] not in stopwords:
			catchy_new.append(catchy[i])
		else:
			continue
	ground_score = 0
	catchy_score = 0
	for i in range (len(ground_new)):
		if ground_new[i] in dict_new:
			ground_score+=dict_new[i]
		else:
			continue
	for i in range (len(catchy_new)):
		if catchy_new[i] in dict_new:
			catchy_score+=dict_new[i]
		else:
			continue
	final = []
	final.append(ground,ground_score,catchy,catchy_score)
	return final