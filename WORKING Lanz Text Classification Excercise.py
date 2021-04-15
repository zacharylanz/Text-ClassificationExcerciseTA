
"""

@author:Zachary Lanz

Instructions for Dr. Young:
    
    1) RUN THE WHOLE PROGRAM
        - You will receive ouput in the following order:
            1. Baseline classification accuracy and 10 most informative features
            2. First preprocessing - second classification accuracy and 10 most informative features
            3. Second preprocessing - third classification accuracy and 10 most informative features
    2) All done!
    
    Please Note: 
        I am only outputting the top 10 most informative features for the sake of not outputting 
        an insane amount of information. Changes between preprocessing 1 and 2 may not be directly 
        visible as stop words were not the most common features. However, the stopword removal does work and 
        can be seen when comparing new_documents and new_documents2.
        
    
"""

import re
import nltk
import random


from nltk.corpus import movie_reviews

##if you want each reivew to be a string:##
documents_raw = [((movie_reviews.raw(fileid), category))
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

## Establishing Baseline/FirstClassification ##

random.shuffle(documents_raw)


all_words = movie_reviews.words()

all_words = []
for word in movie_reviews.words():
    all_words.append(word.lower())
    
    
# use nltk method FreqDist to convert (destructively) all_words into list of word frequencies
all_words = nltk.FreqDist(all_words)

# inspect all_words - for testing onlny 
#print(all_words)

# print 15 most common words - for testing only
#print(all_words.most_common(15))

# the top 3,000 most common words
word_features = list(all_words.keys())[:3000]

# For any document, check to see if word features are present
def find_features(documents_raw):
    words = set(documents_raw)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# feature set for one review example
#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

# Create feature sets for all reviews and associate with review cagtegory
featuresets = [(find_features(rev), category) for (rev, category) in documents_raw]

# Create training set from first 1900 reviews in unshuffled set
training_set = featuresets[:1900]

# Create testing set from remaining reviews in unshuffled set
testing_set = featuresets[1900:]

# create and train classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier 1 Accuracy:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(10)


##Set functions/Variables##

def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens            

#Does a word tokenization of the input text
def tokenize_text(text):
    #our function calls (uses) functions from the nltk library we imported
    words = nltk.word_tokenize(text)
    #print(words)
    return words

    
First_doc=documents_raw[0]
First_review=First_doc[0]

Tokenized_text=tokenize_text(First_review)

def remove_special_characters(string):
    # remove any leading or trailing spaces using string.strip method
    string = string.strip()
    
    # create a pattern for anything but alpha-numeric characters  ^ = not
    PATTERN = r'[^a-zA-Z0-9 ]'
    filtered_string = re.sub(PATTERN, r'', string)
    return filtered_string
#remove_special_characters("this is my test: $%^&%#@")


## Preprocessing 1 ##


def preprocess_document_tuple (documents):

        new_documents=[]
        for document in documents:
            # debug print
            #print(document[0])
        
            #note you may need to coerce document[0] to a string => str(document[0])
            #new_document = str(document[0])
            #DO YOUR STUFF HERE!
         
            
            #Remove Special Characters
           
            #FOR TESTING PURPOSES ONLY:
            #random.shuffle(documents_raw)
            
            new_document=remove_special_characters(document[0])


            new_document=tokenize_text(new_document)
            
            
            #add post-processed document to new_document list for passing to classification process.
            new_documents.append((new_document, document[1]))
        
            
        # debug print
        #print(new_documents)
        
        #print (len(new_documents))
        
        return new_documents

new_documents=preprocess_document_tuple(documents_raw)

## CLASSIFIER POST PREPROCESSING 1 ##

featuresets1 = [(find_features(rev), category) for (rev, category) in new_documents]

# Create training set post preprocessing1
new_training_set1 = featuresets1[:1900]

# Create testing set post preprocessing1
new_testing_set1 = featuresets1[1900:]

classifier = nltk.NaiveBayesClassifier.train(new_training_set1)

print("Preprocessed Classifier 2 Accuracy:",(nltk.classify.accuracy(classifier, new_testing_set1))*100)

classifier.show_most_informative_features(10)


##PREPROCESSING 2##


def preprocess_document_tuple2 (documents):
        
        new_documents2=[]
        for document in documents:
            # debug print
            # print (len(document[1]))
            
        
            #note you may need to coerce document[0] to a string => str(document[0])
            #new_document = str(document[0])
            #DO YOUR STUFF HERE!
            
            new_document=remove_stopwords(document[0])
    
            new_documents2.append((new_document, document[1]))
 
            
            # debug print
            # print(new_documents2)
       
        return new_documents2
        
    
new_documents2=preprocess_document_tuple2(new_documents)

## CLASSIFIER POST PREPROCESSING 2 ##

featuresets2 = [(find_features(rev), category) for (rev, category) in new_documents2]

# Create training set post preprocessing2
new_training_set2 = featuresets2[:1900]

# Create testing set post preprocessing2
new_testing_set2 = featuresets2[1900:]

classifier = nltk.NaiveBayesClassifier.train(new_training_set2)

print("Preprocessed Classifier 3 accuracy:",(nltk.classify.accuracy(classifier, new_testing_set2))*100)

classifier.show_most_informative_features(10)


