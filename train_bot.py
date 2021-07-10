import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

words=[]
classes=[]
documents=[]
ignore_words=['?','!']
data_file = open("intents.json").read()
intents = json.loads(data_file)

for intent in intents['intents']:
	for pattern in intent['patterns']:
		#tokenize here
		w=nltk.word_tokenize(pattern)
		print('Token is: ()',format(w))
		words.extend(w)
		documents.append((w, intent['tag']))
		# adding the tags to the classes lists
	if intent['tag'] not in classes:
		classes.append(intent['tag'])

	#List of final items
	#print('Words list is: {}'.format(words))
	#print("Docs are: {}".format(documents))
	#print('classes are: {}'.format(classes))

	words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
	words = list(set(words))
	#print(words)
	pickle.dump(words,open('words.pkl', 'wb'))
	pickle.dump(classes, open('classes.pkl', 'wb'))

	training = []
	output_empty = [0]*len(classes)
	# [0,0,0,0,0,0]
	for doc in documents:
		bag = []
		pattern_words = doc[0]
		pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
		#print('Current Pattern Words: {}'.format(pattern_words))

		for w in words:
			bag.append(1) if w in pattern_words else bag.append(0)

		#print('Current Bag: {}'.format(bag))

		output_row = list(output_empty)
		output_row[classes.index(doc[1])] = 1
		#print('Current Output: {}'.format(output_row))

		training.append([bag, output_row])



