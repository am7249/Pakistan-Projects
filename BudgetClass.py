import numpy as np
import matplotlib.pyplot as plt
import os
import string
import random

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk import pos_tag
from nltk.util import ngrams 
from nltk.stem.snowball import SnowballStemmer

from palettable.colorbrewer.diverging import Spectral_11
from matplotlib.colors import makeMappingArray
from palettable.colorbrewer.qualitative import Dark2_8, Set3_12, Paired_12_r
from palettable.colorbrewer.sequential import Reds_9

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from gensim.summarization.summarizer import summarize
from collections import Counter
from textblob import TextBlob

from PIL import Image

import re



class Budget:
    def __init__(self, file_path, name, flag):
        self.name = name
        self.remove_stop_words = flag
        self.text = self.read_file(file_path)
        self.tokens = self.tokenize(self.text)
        self.unique_words = list(set(self.tokens))
        self.sentence_tokens = self.token_sentences(self.text)
        self.processed_text = self.preprocess(self.tokens)
        self.stemmed_list = self.stemmer(self.processed_text)
        self.word_frequency = self.entire_word_frequency(self.processed_text)
        self.nouns = self.preprocess(self.get_nouns(self.tokens))
        self.verbs = self.preprocess(self.get_verbs(self.tokens))
        
    def read_file(self, file_path):
        '''Input: file path
           Output: text in the file'''
        
        assert os.path.exists(file_path), "File not found at: "+str(file_path)
        
        f = open(file_path,'r')    
        text = f.read()
        f.close()
        
        return text
    
    def tokenize(self, text):
        '''Input: text
           Output: tokens in the text'''
        return nltk.word_tokenize(text)
    
    
    def token_sentences(self, text):
        '''Input: text
           Output: sentence tokens
        '''
        sent_text = nltk.sent_tokenize(text)
        return sent_text
    
    
    def preprocess(self, tokens):
        '''Input: tokens
           Function: turns tokens into lower case,
                     removes the puncuation,
                     keeps only alphanumeric tokens
                     removes stop words
           Output: processed text'''
        
        tokens = [w.lower() for w in tokens] #turns to lower case
        
        table = str.maketrans('', '', string.punctuation) #removes punctuation
        stripped = [w.translate(table) for w in tokens]
        
        words = [word for word in stripped if word.isalpha()] #keeps only alphanumeric tokens
        
        if self.remove_stop_words:
            stop_words = set(stopwords.words('english')) #removes stop words
            words = [w for w in words if not w in stop_words]
        
        return words
    
    def stemmer(self, processed_text):
        '''Input: processed text
           Output: tokens after stemming
        '''
        st = RSLPStemmer()
        #st = SnowballStemmer("english")
        stemmed_list = set(st.stem(token) for token in processed_text)
        return stemmed_list
    
    def entire_word_frequency (self, processed_text):
        '''Input: processed tokens
           Output: word count
        '''
        word_frequency = nltk.FreqDist(processed_text)
        return word_frequency
    
    def compute_word_frequency (self, n):
        '''Input: n
           Output: n most common words
        '''
        word_frequency = nltk.FreqDist(self.processed_text)
        mostcommon = word_frequency.most_common(n)
        return mostcommon    
    
    def get_nouns(self, tokens):
        tags = nltk.pos_tag(tokens)
        nouns = [word for word, pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
        return nouns
    
    def get_verbs(self, tokens):
        tags = nltk.pos_tag(tokens)
        verbs = [word for word, pos in tags if (pos == 'VB' or pos == 'VBG' or pos == 'VBD' or pos == 'VBN'
                                               or pos == 'VBP' or pos == 'VBZ')]
        return verbs
    
    
    
def create_wordcloud(text, title = None, maximum_words = 100):        
    def color_func_1(word, font_size, position, orientation, random_state=None, **kwargs):
        return tuple(Dark2_8.colors[random.randint(0,7)])
    
    def grey_color_func_1(word, font_size, position, orientation, random_state=None, **kwargs):
        return "hsl(0, 0%%, %d%%)" % random.randint(10, 50)
    
    def color_func_2(word, font_size, position, orientation, random_state=None, **kwargs):
        return tuple(Reds_9.colors[random.randint(2,8)])
    
    stop_words = set(list(STOPWORDS) + ['pakistan','instrument', 'people', 'party', 'price', 'government', 'per', 
                    'cent', 'will', 'parliamentarians','rs','i', 'ii', 'iv', 'v', 
                     'vi','vii', 'ensure', 'right', 'provide', 'increase', 'decrease', 'budget','budget',
                     'Mr Speaker', 'proposed', 'shall', 'present', 'formulated', 'section',
                     'muslim','year', 'mr','speaker', 'madam', 'million', 'billion', 'tax',
                     'sector', 'rate', 'speech', 'page', 'country', 'may', 'lo', 'po', 'al','de',
                     'le','dre', 'person', 'persons', 'years'])
    
    font_path = "/Users/macbook/Downloads/amatic-sc/AmaticSC-Bold.ttf"
    #font_path = "/Users/macbook/Downloads/advent-pro/AdventPro-Light.ttf"
    gradient_orientation = "h"
    icon = Image.open("img/flag_img2.png").convert("RGBA")
    building_mask = Image.new("RGBA", icon.size, (255,255,255))
    building_mask.paste(icon,icon)
    mask_wordcloud = np.array(building_mask)
    

    wordcloud = WordCloud(font_path=font_path, stopwords=stop_words, background_color="white", max_words=2000,
                          mask=mask_wordcloud, max_font_size=300, random_state=42).generate_from_text(text)

    
    fig = plt.figure(1, figsize=(15, 10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=30)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud.recolor(color_func=color_func_1, random_state=3), interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    
def summarize_text(text):
    '''
    '''
    print (summarize(text, ratio = 0.02))
        
        
def sentiment_analysis(sentence_tokens, name):
    '''
    Given sentence tokens, calculates polarity and subjectivity of each sentence in the document,
        and plots graphs of polarity and subjectivity
    Input:
        sentence_tokens: list of strings
        name: the name of each party (string)
    '''
    polarity = []
    subjectivity = []
    for sentence in sentence_tokens:
        s = TextBlob(sentence)
        polarity.append(s.sentiment[0])
        subjectivity.append(s.sentiment[1])
        
    plt.plot(polarity)
    plt.xlabel('Sentences across budget')
    plt.ylabel('Polarity')
    plt.title(' Budget by '+ name)
    plt.show()


    plt.plot(subjectivity, 'C7')
    plt.xlabel('Sentences across budget')
    plt.ylabel('subjectivity')
    plt.title(' Budgey by '+ name)
    plt.show()
    
def document_stats(documents):
    for budget in documents:
        print(budget.name)
        
        #sentiment analysis
        s = TextBlob(" ".join(budget.processed_text))
        print("polarity: ", round(s.sentiment[0],2)) #between -1 (negative) and 1 (positive)
        print("subjectivity: ", round(s.sentiment[1],2)) #between 0 (very objective) and 1 (very subjective)
        
        print(budget.compute_word_frequency(20))
        print('\n')