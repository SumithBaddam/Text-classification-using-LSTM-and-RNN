import nltk
from nltk.corpus import brown
import sys
import pandas as pd
from nltk.corpus import stopwords
from collections import defaultdict
from collections import Counter
import math
from pymongo import MongoClient


stop = set(stopwords.words('english'))

brown_train = brown.tagged_sents(categories='news')
regexp_tagger = nltk.RegexpTagger(
    [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
     (r'(-|:|;)$', ':'),
     (r'\'*$', 'MD'),
     (r'(The|the|A|a|An|an)$', 'AT'),
     (r'.*able$', 'JJ'),
     (r'^[A-Z].*$', 'NNP'),
     (r'.*ness$', 'NN'),
     (r'.*ly$', 'RB'),
     (r'.*s$', 'NNS'),
     (r'.*ing$', 'VBG'),
     (r'.*ed$', 'VBD'),
     (r'.*', 'NN')
])

unigram_tagger = nltk.UnigramTagger(brown_train, backoff=regexp_tagger)
bigram_tagger = nltk.BigramTagger(brown_train, backoff=unigram_tagger)


cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
cfg["JJ+JJ"] = "JJ"
cfg["JJ+NN"] = "NNI"


class NPExtractor(object):
    def __init__(self, sentence):
        self.sentence = sentence
    def tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens
    def normalize_tags(self, tagged):
        n_tagged = []
        for t in tagged:
            if t[1] == "NP-TL" or t[1] == "NP":
                n_tagged.append((t[0], "NNP"))
                continue
            if t[1].endswith("-TL"):
                n_tagged.append((t[0], t[1][:-3]))
                continue
            if t[1].endswith("S"):
                n_tagged.append((t[0], t[1][:-1]))
                continue
            n_tagged.append((t[0], t[1]))
        return n_tagged
    # Extract the main topics from the sentence
    def extract(self):
        tokens = self.tokenize_sentence(self.sentence)
        tags = self.normalize_tags(bigram_tagger.tag(tokens))
        #print tags
        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = "%s+%s" % (t1[1], t2[1])
                value = cfg.get(key, '')
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = "%s %s" % (t1[0], t2[0])
                    pos = value
                    tags.insert(x, (match, pos))
                    break
        matches = []
        for t in tags:
            #if t[1] == "NNP" or t[1] == "NNI":
            if t[1] == "NNP" or t[1] == "NNI" or t[1] == "NN":
                matches.append(t[0])
        return matches

 


def main():
	all_words_counts = {}
	mtkg_names_words = {}
	d = defaultdict(list)
	data = pd.read_csv('Feature_asso.csv')
	#print(data['SUMMARY'])
	print("Data Loaded...")
	#sentence = "Can you suggest me good books"
	#sentence = "N/A"
	#sentence = sys.argv[1]
	feature_names = data['FEATURE_ID']
	all_mktg_names = data['MKTG_NAME']
	all_summary_sentences = data['SUMMARY']
	all_description_sentences = data['FULL_DESC']
	#print(len(all_summary_sentences))
	print("Intent for MKTG Names")
	for i in range(0, len(all_mktg_names)):
		#print(all_summary_sentences[i])
		if(pd.isnull(all_mktg_names[i]) == False and all_mktg_names[i] != "None"):
			np_extractor = NPExtractor(all_mktg_names[i])
			if(i%100 ==0):
				print(i)
			#print np_extractor
			results = np_extractor.extract()
			#if(i<100):
			#	print(results)
			for word in results:
				if(feature_names[i] == 3559):
					print(word)
				if((word != 'N/A') and (word not in stop)):
					#d[word].append(feature_names[i])
					if(word not in d[feature_names[i]]):
						d[feature_names[i]].append(word)
					if ((word not in all_words_counts)):
						#d[feature_names[i]].append(word)
						if(feature_names[i] == 1993):
							print("inside...")
							print(word)
						all_words_counts[word] = 1
						mtkg_names_words[word] = 1
					else:
						all_words_counts[word] = all_words_counts[word] + 1
						mtkg_names_words[word] += 1
	#mtkg_names_words = all_words_counts
	print("Intent for Summary sentences")
	for i in range(0, len(all_summary_sentences)):
		#print(all_summary_sentences[i])
		if(pd.isnull(all_summary_sentences[i]) == False and all_summary_sentences[i] != "None"):
			np_extractor = NPExtractor(all_summary_sentences[i])
			if(i%100 ==0):
				print(i)
			#print np_extractor
			results = np_extractor.extract()
			for word in results:
				if((word != 'N/A') and (word not in stop)):
					#d[word].append(feature_names[i])
					if(word not in d[feature_names[i]]):
						d[feature_names[i]].append(word)
					if ((word not in all_words_counts)):
						#d[feature_names[i]].append(word)
						all_words_counts[word] = 1
					else:
						all_words_counts[word] = all_words_counts[word] + 1
	print("Intent for Full Descriptions")
	for i in range(0, len(all_description_sentences)):
		#print(all_summary_sentences[i])
		if(pd.isnull(all_description_sentences[i]) == False and all_description_sentences[i] != "None"):
			np_extractor = NPExtractor(all_description_sentences[i])
			if(i%100 ==0):
				print(i)
			results = np_extractor.extract()
			for word in results:
				if((word != 'N/A') and (word not in stop)):
					#d[word].append(feature_names[i])
					if(word not in d[feature_names[i]]):
						d[feature_names[i]].append(word)
					if ((word not in all_words_counts)):
						#d[feature_names[i]].append(word)
						all_words_counts[word] = 1
					else:
						all_words_counts[word] = all_words_counts[word] + 1
	if("MI" in all_words_counts):
		all_words_counts.pop("MI")
	print("########################################")
	print(len(mtkg_names_words))
	return(all_words_counts, mtkg_names_words, d)

def get_top_words(all_words_counts, mtkg_names_words, all_num, mtkg_num):
	t_m = sorted(mtkg_names_words.iteritems(), key=lambda x:-x[1])[:mtkg_num]
	t_a = sorted(all_words_counts.iteritems(), key=lambda x:-x[1])[:all_num]
	#for x in t:
	#	print "{0}: {1}".format(*x)
	new_dict = {}
	for x in t_m:
		new_dict[x[0]] = all_words_counts[x[0]]
	#print(len(new_dict))
	for x in t_a:
		if(x[0] not in new_dict):
			#i+=1
			#print(x[0])
			new_dict[x[0]] = all_words_counts[x[0]]
#	print(i)
	return new_dict


def remove_duplicates(word_feature_ids):
	for i in word_feature_ids:
		word_feature_ids[i] = list(set(word_feature_ids[i]))
	return word_feature_ids


def getFeatureID_for_topwords(word_feature_ids, top_words):
	final_words_features = defaultdict(list)
	for a in word_feature_ids:
		print(a)
		for b in word_feature_ids[a]:
			if(b in top_words):
				#if(!(len(final_words_features[a]))):
				final_words_features[a].append(b)
				#else:
				#	final_words_features[a] = [b]
			else:
				for c in top_words:
					if (c in b):
						final_words_features[a].append(b)
	'''
	for i in word_feature_ids.keys():
		print(i)
		for a in word_feature_ids[i]:
			if a not in top_words:
				word_feature_ids[i].remove(a)
	'''
	return final_words_features


def finding_partial_words(all_words_counts):
	#t["MIB] = 506+108=614
	all_words = all_words_counts.keys()
	for a in all_words:
		if(len(a) < 3):
			#print(a)
			all_words.remove(a)
		elif(len(a) > 1000):
			all_words.remove(a)	
	all_words.sort()
	#print(all_words)
	if('FC' in all_words):
		print("there")
	for a in all_words:
		if(len(a) == 2):
			print("Len is 2")
			#print(a)
			all_words.remove(a)
	if('FC' in all_words):
		print("STill there")
	#print(all_words_counts['RFC'])
	final_dict = {}
	similar_words = defaultdict(list)
	#k=0
	#print(all_words)
	for i in range(0, len(all_words)):
		if(all_words[i] not in final_dict):
			#print(all_words[i])
			#k+=1
			#if(all_words[i] == 'FC'):
			#	print('In FC')
			#if(all_words[i] == 'RFC'):
			#	print("Here")
			#	print(all_words_counts[all_words[i]])
			final_dict[all_words[i]] = all_words_counts[all_words[i]]
			#if(all_words[i] == "DMVPN"):
			#	print(final_dict[all_words[i]])
			for j in range(i+1, len(all_words)):
				if(all_words[i] in all_words[j]):
					#if(all_words[i] == "DMVPN"):
					#	print(final_dict[all_words[i]], all_words[j], all_words_counts[all_words[j]])
					final_dict[all_words[i]] += all_words_counts[all_words[j]]
					#if(all_words[j] == "RFC"):
					#	print("It's here", all_words[i], all_words[j])
					final_dict[all_words[j]] = 0
					similar_words[all_words[i]].append(all_words[j])
					#print(all_words[i], all_words[j], final_dict[all_words[i]], all_words_counts[all_words[j]], final_dict[all_words[i]])
	#print(k)
	print(final_dict['RFC'])
	#remove all words with zero value
	n={}
	n = {key: value for key, value in final_dict.items() if value != 0}
	return n, similar_words

def get_features_topWords(word_feature_ids, t, rev_s):
	new_feature_maps = defaultdict(list)
	top_words = t.keys()
	for feature in word_feature_ids:
		print(feature)
		for word in word_feature_ids[feature]:
			split_word = word.split()
			status=0
			for w in split_word:
				#if w not in top_words:
				#	status=1
				if w in top_words:
					if(w in s):
						new_feature_maps[feature].append(w)
					else:
						if(w in rev_s):
							new_feature_maps[feature].append(rev_s[w])
						else: #Eg: ATCP is in top and it doesn't have similar words
							new_feature_maps[feature].append(w)
			#if(status==1):
			#	word_feature_ids[feature].remove(word)
	return 	new_feature_maps


def reverse_map_similarWords(s):
	reverse_s={}
	for word in s:
		for word_list in s[word]:
			#for w in word_list:
			reverse_s[word_list] = word
	return reverse_s


def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)
    


def finding_clusters(new_feature_maps):
	h={}
	for i in new_feature_maps:
		print(i)
		counterA = Counter(new_feature_maps[i])
		if(len(counterA) != 0):
			for j in range(i+1, len(new_feature_maps)):
				if(len(new_feature_maps[j]) > 0):
					counterB = Counter(new_feature_maps[j])
					#print("here")
					if(counter_cosine_similarity(counterA, counterB) > 0.5):
						#print(counterA, counterB, i, j)
						#print("here2")
						h[i] = j
						#print("Hashed")
						break
	return h

def group_clusters(h):
	final_list = {}
	l=[]
	m = h.keys()
	for i in m:
		#print("i is ", i)
		l.append(i)
		j=h[i]
		while(1):
			l.append(j)
			if(j in h):
				#print("j is ", j)
				#print(m[:6000])
				if(j in m):
					#print("NOT")
					m.remove(j)
				j=h[j]
			else:
				break
	
		#print(l)
		print(i)
		final_list[i]=l
		l=[]
	return final_list
#length of one is 249...

def getting_words_wordcloud(final_list, new_feature_maps):
	all_words = defaultdict(list)
	for feat_id in final_list:
		for sim_id in final_list[feat_id]:
			all_words[feat_id].append(' '.join(new_feature_maps[sim_id]))
	return(all_words)



def write_to_mongo(final_json):
	client = MongoClient('sjc-qddts-bi1:27017')
	db = client.buganalytics
	


if __name__ == '__main__':
	a, m, word_feature_ids = main()
	#word_feature_map = remove_duplicates(word_feature_ids) #Needed if we use word to feature IDs mapping.
	###Take all the MKTG_NAME words...they are important...and a few more words from overall thing.
	###Now we got the words we should consider
	t = get_top_words(a, m, 10000, 1000)#Lets take the top few words
	###For each feature, if split(word) is not in 't' then remove.
	#feature_topwords_map = getFeatureID_for_topwords(word_feature_ids, t.keys()) #NEED TO WORK...
	#final_feature_word = get_features_topWords(word_feature_ids, t)
	###Club the partially same words
	f, s = finding_partial_words(t)
	rev_s = reverse_map_similarWords(s)
	new_feature_maps_with_dups = get_features_topWords(word_feature_ids, t, rev_s)
	###Now we know the top words, their similar words, the feature IDs that contain these words...
	new_feature_maps = remove_duplicates(new_feature_maps_with_dups)
	###Cosine similarity for the list of lists
	###For each ID we will find first occurrence of another similar ID and store that. Associative property holds.
	###http://stackoverflow.com/questions/14720324/compute-the-similarity-between-two-lists
	#counterA = Counter(new_feature_maps[3559])
	#counterB = Counter(new_feature_maps[3509])
	#score = counter_cosine_similarity(counterA, counterB)
	hashed_map = finding_clusters(new_feature_maps)
	final_json = group_clusters(hashed_map)
	all_words_list = getting_words_wordcloud(final_list, new_feature_maps)
	###For each feature with every other feature...we find how much mapping is there(i.e how many top words (from the list) are same)
