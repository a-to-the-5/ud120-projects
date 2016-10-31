from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from nltk.probability import FreqDist
from nltk import word_tokenize
import re
import pickle

d = path.dirname(__file__)

source_dir = 'C:/Users/aahmed/Downloads/enron_mail_20150507/maildir/'
print source_dir

# tokens = []
# for subdir in os.listdir(source_dir):
#     print subdir
#     print len(tokens)
#     if os.path.isdir("%s/%s/inbox/" % (source_dir, subdir)):
#         for f in os.listdir("%s/%s/inbox/" % (source_dir, subdir)):
#             if re.match('\d+_$', f):
#                 path = "%s/%s/inbox/%s" % (source_dir, subdir, f)
#                 with open(path, 'r') as myfile:
#                     text = myfile.read()
#                     tokens.extend(word_tokenize(text.lower()))
# print "tokens: ", len(tokens)
# with open("tokens.pkl",'wb') as f:
#     pickle.dump(tokens, f)
with open("tokens.pkl",'rb') as f:
    tokens = pickle.load(f)
freq = FreqDist([w for w in tokens if
                 len(w) > 7 and re.match('[a-z]+$', w) and w not in ['this', 'that', 'these', 'from', 'with', 'will', 'your', 'also', 'http', 'which', 'have', 'there']])
wordcloud = WordCloud().generate_from_frequencies(freq.most_common(100))
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
