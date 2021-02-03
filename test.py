
import feature_extractor as fe
import re
import sys
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')
import numpy as np
from collections import Counter
from nltk.util import ngrams



punct = '.,:;-\'\"(!?'
punct_list = list(punct)

res = " ".join(("[{}]".format(*i) for i in punct_list)) 
print(res)


# new_list = [" ".join("'{}'".format(words) for words in seq)  for seq in original_list]

# new_list = [" ".join(words if not " " in words and i == 0 else "({})".format(words) for i, words in enumerate(seq))  for seq in original_list]