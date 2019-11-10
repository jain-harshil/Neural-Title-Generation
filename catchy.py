import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import random

def count(string):
	return sum(1 for c in string if c.isupper())

def catchy(t,text,key):
	s = t.split()
	r = []
	flag = 0
	r.append(s[0])
	for i in range (1,len(s)):
		if s[i] == s[i-1]:
			continue
		else:
			r.append(s[i])
	for i in range (len(r)):
		r[i] = r[i][0].upper()+r[i][1:]
	stop_words = set(stopwords.words('english')) 
	text = re.sub(r'[^A-Za-z\s]',"",text)
	text = re.sub(r'https\S+', "", text)
	text = text.lstrip()
	text = text.strip()
	list1 = word_tokenize(text)
	fil_sent = [w for w in list1 if not w in stop_words]

	final = []
	c = {}
	for i in fil_sent:
		if i[1:] != i[1:].lower() and count(i)>3:
			final.append(i)
			if i not in c:
				c[i]=1
			else:
				c[i] += 1
	ans = ""
	if len(c)!=0:
		max_value = max(c.values())
		last = [k for k,v in c.items() if v == max_value]
		length = []
		if len(last)>=1:
			for i in range (len(last)):
				if last[i] == last[i].upper():
					flag = 1
					ans = last[i]
					length.append(len(last[i]))
				else:
					length.append(len(last[i]))
			if ans == "":
				flag = 1
				ans = last[length.index(max(length))]
	final = []
	if ans == "":
		if len(key)>0:
			flag = 2
			pre = re.split('-|;|,|—|:',key)
			for i in range (len(pre)):
				if pre[i] != "" and pre[i] !="keywords" and pre[i]!="Keywords" and pre[i]!="KEYWORDS":
					final.append(pre[i])
			for i in range(len(final)):
				final[i] = final[i].lstrip()
				final[i] = final[i].rstrip()
			print(final)
			z = random.randrange(0,len(final))
			ans = final[z]
			ans = ans.replace(" ","")

	ans1 = ""
	for i in range (len(r)):
		if i == 0:
			ans1 = ans1+r[i]
		else:
			ans1 = ans1+" "+r[i]
	if flag == 2:
		ans1 = "#"+ans+": "+ans1
	elif flag == 0:
		ans1 = ans1
	else:
		ans1 = ans+": "+ans1
	return (ans1)

abstract = ". Digital advancement in scholarly repositories has led to the emergence of a large number of open access predatory publishers that charge high article processing fees from authors but fail to provide nec-essary editorial and publishing services. \n Identifying NEC-\n essary and blacklisting such publishers has remained a research challenge due to the highly volatile scholarly publishing ecosystem. This paper presents a data-driven ap-proach to study how potential predatory publishers are evolving and bypassing several regularity constraints. We empirically show the close resemblance of predatory publishers against reputed publishing groups. In addition to verifying standard constraints, we also propose distinc-tive signals gathered from network-centric properties to understand this evolving ecosystem better. To facilitate reproducible research, we shall make all the codes and the processed dataset available in the public domain. Keywords: Predatory Journals, Publication Ethics, Open Access and Digital Library"
abstract1 = "We introduce SinGAN, an unconditional generative model that can be learned from a single natural image. Our model is trained to capture the internal distribution of patches within the image, and is then able to generate high quality, diverse samples that carry the same visual content as the image. SinGAN contains a pyramid of fully convolutional GANs, each responsible for learning the patch distribution at a different scale of the image. This allows generating new samples of arbitrary size and aspect ratio, that have significant variability, yet maintain both the global structure and the fine textures of the training image. In contrast to previous single image GAN schemes, our approach is not limited to texture images, and is not conditional (i.e. it generates samples from noise). User studies confirm that the generated samples are commonly confused to be real images. We illustrate the utility of SinGAN in a wide range of image manipulation tasks"
keywords = "Keywords: Predatory Journals, Publication Ethics, Open Access and Digital Library"

print(catchy("automatic extraction extraction clinical texts using web based approach approach approach approach",abstract1,keywords))