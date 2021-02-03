# extract features from list of text instances based on configuration set of features

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk, re
nltk.download('averaged_perceptron_tagger')
import numpy as np
from collections import Counter
from nltk.util import ngrams

def function_words(texts):
	bow = []
	header = stopwords.words('english')
	for text in texts:	#get stopwords counts for each text
		counts = []
		tokens = nltk.word_tokenize(text)
		for sw in stopwords.words('english'):
			sw_count = tokens.count(sw)
			normed = sw_count/float(len(tokens))
			counts.append(normed)
		bow.append(counts)
	bow_np = np.array(bow).astype(float)
	return bow_np, header	

def syntax(texts):
	common_pos_tags = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS','CC','PRP','VB','VBG']
	tag_counts = []

	for text in texts:
		tokens = nltk.word_tokenize(text)
		tags = nltk.pos_tag(tokens)
		temp = []
		counts = Counter( tag for word,  tag in tags )
		for each_tag in common_pos_tags:
			common_tag = []
			common_tag = counts[each_tag]
			temp.append(common_tag)
		tag_counts.append(temp)
	tag_np = np.array(tag_counts).astype(float)

	return tag_np, common_pos_tags


def lexical(texts):

	common30unigrams = []
	token_freq_list = []
	all_text = ""

	for text in texts:  # get all essays and get the most frequent tokens
		all_text = all_text + " " + text
	
	no_punc = re.sub(r'[^\w\s]', "", all_text)
	tokens = nltk.word_tokenize(no_punc)
	stop_words = set(stopwords.words('english'))
	no_stop_tokens = [w for w in tokens if not w in stop_words] 
	unigram = dict(Counter(no_stop_tokens).most_common(30))

	res = " ".join(("{}".format(*i) for i in unigram.items())) 
	# returned keys of the top 30 unigrams in clearer format
	common30unigrams = nltk.word_tokenize(res)
	# converted the words back into a List from Counter 

	for text in texts:
		no_punc = re.sub(r'[^\w\s]', "", text)
		tokens = nltk.word_tokenize(no_punc)
		no_stop_tokens = [w for w in tokens if not w in stop_words] 
		tokens = nltk.word_tokenize(text)
		temp = []
		counts = Counter( tokens )
		for each_unigram in common30unigrams:
			token_freq = []
			token_freq = counts[each_unigram]
			temp.append(token_freq)
		token_freq_list.append(temp)
	token_np = np.array(token_freq_list).astype(float)

	return token_np, common30unigrams


def punctuation(texts):

	punct = '.,:;-\'\"(!?'
	punct_list = list(punct)

	punct_counts = []

	for text in texts:
		tokens = re.sub(r'[\w\s]', "", text)
		punct_tokens = nltk.word_tokenize(tokens)

		temp = []

		counts = Counter(punct_tokens)
		for each_punct in punct_list:
			common_punct = []
			common_punct = counts[each_punct]
			temp.append(common_punct)
		punct_counts.append(temp)
	punct_np = np.array(punct_counts).astype(float)

	res = " ".join("{}".format(*i) for i in punct_list)

	return punct_np, list(res)


def complexity(texts):
	complex = []
	headers = ['average # of char per word', 'TTR', 'average # of word per sentence', 'long words count']

	for text in texts:
		counts = []
		long_word = 0
		words = nltk.word_tokenize(text)
		avg_char = sum(len(word) for word in words) / len(words)
		counts.append(avg_char)

		ttr = (len(Counter(words)))/len(words)
		counts.append(ttr)

		for word in words:
			if (len(word) >= 6):
				long_word += 1

		sents = nltk.sent_tokenize(text)

		avg_word = sum(len(sent.split()) for sent in sents) / len(sents)
		counts.append(avg_word)
		counts.append(long_word)

		complex.append(counts)

	complex_np = np.array(complex).astype(float)

	return complex_np, headers


def extract_features(texts, conf):
	features = []
	headers = []

	if 'function_words' in conf:
		f,h = function_words(texts)
		features.append(f)
		headers.extend(h)

	if 'syntax' in conf:
		f,h = syntax(texts)
		features.append(f)
		headers.extend(h)

	if 'lexical' in conf:
		f,h = lexical(texts)
		features.append(f)
		headers.extend(h)

	if 'punctuation' in conf:
		f,h = punctuation(texts)
		features.append(f)
		headers.extend(h)

	if 'complexity' in conf:
		f,h = complexity(texts)
		features.append(f)
		headers.extend(h)

	all_features = np.concatenate(features,axis=1)
	return all_features, headers
