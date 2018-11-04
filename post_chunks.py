import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import h5py
from sklearn.feature_extraction.text import TfidfVectorizer
import ssl
from nltk.corpus import stopwords

def download_nltk():
    
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('averaged_perceptron_tagger')   
    nltk.download('punkt')



def get_tokens(data):
    items = ['NN', 'JJ', 'VB', 'RB']
    token_list=[]
    word_list = []
    for d in data:
        token = nltk.word_tokenize(d)
        x = nltk.pos_tag(token)       
        token_list.append(x)
        grammar = lambda pos: pos[:2] in items
        words = [t for t,pos in x if grammar(pos)]
        word_list.append(words)

    return token_list, word_list
    

def get_phrases(token_list):
    noun1 = "NP1: {<JJ.*>+<NN.*>}"
    noun2 = "NP2: {<NN.*>+<NN.*>}"
    verb1 = "VP1: {<VB.*>+<NN.*>}"
    verb2 = "VP2: {<VB.*>+<DT>+<NN.*>}"
    verb3 = "VP3: {<VB.*>+<TO>+<DT><NN.*>}"
    verb4 = "VP4: {<VB.*>+<TO>+<NN.*>}"
    verb5 = "VP5: {<VB.*>+<IN>+<DT>+<NN.*>}"
    adj1 = "ADJP1: {<JJ.*>+<JJ.*>}"
    adj2 = "ADJP2: {<RB.*>+<JJ.*>}"
    adv = "ADVP: {<RB.*>+<VB.*>}"
    #prep1 = "PREP1: {<IN>+<DT>+<NN.*>}"
    #prep2 = "PREP2: {<IN>+<NN.*>}"
    grammar = [noun1, noun2, verb1, verb2, verb3, verb4, adj1, adj2, adv] #prep1, prep2]
    phrase_list = []
    for tokens in token_list:
        phrase = []
        for g in grammar:
            cp = nltk.RegexpParser(g)
            result=cp.parse(tokens)
            phrase.append(result)
            #print(type(result))
            #print([w for w,pos in result.pos() if (pos == 'PREP')])

        grammar1 = ['NP1', 'NP2', 'VP1', 'VP2', 'VP3', 'VP4', 'VP5', 'ADJP1', 'ADJP2', 'ADVP'] #, 'PREP1', 'PREP2']
        ps = []
        Ps = [(p.subtrees(filter=lambda x: x.label() in grammar1)) for p in phrase]#got subtrees

        for i in range(0,len(Ps)):
            ph = [p for p in Ps[i]]
            ps.extend([w for w in ph])

        p = []
        p.extend([ps[i] for i in range(0,len(ps))])
        phrases = []
        for i in range(0,len(p)):
            phrases.append(' '.join([w for w, t in p[i].leaves()]))

        phrase_list.append(phrases)
    print(len(phrase_list))
    #print(phrase_list[660])
    return phrase_list


def stemming(word_list):
    stem = []
    lists = []
    ps = PorterStemmer()
    for words in word_list:
        for w in words:
            stem.append(ps.stem(w))
        lists.append(stem)
    return lists


def clean_text(text_list):
    """
    Take a text_list as an argument and
    replace numbers
    return a cleaned text list.
    Remove parsing noise.
    """
    with open('Round3name.txt', 'r') as file:  
        for n in file:
            names = re.sub("[^a-zA-Z\- ]", "", n)
            names = names.split()
    
    names1 = [n.capitalize() for n in names]
    names2 = [n.upper() for n in names]
    
    names.extend(names1)
    names.extend(names2)

    clean_text = []
    clean = []
    for text in text_list:
        t = re.sub("\n", " ", text) #remove newline character
        t = re.sub("\xa0", " ", t) #remove noise from parsing
        t = re.sub("\ufeff", " ", t) #remove noise from parsing
        t = re.sub("[0-9]+", "", t) #replace numbers
        t = re.sub("Subscribe", "", t)
        t = re.sub("Click to share", "", t)
        t = re.sub("bastard", "", t)
        t = re.sub("Mr.","",t)
        t = re.sub("Ms.","",t)
        t = re.sub("Mrs.","",t)
        t = re.sub("St.", "", t)
        t = re.sub("(Politics|Business|World|National|Sport|Entertainment|Lifestyle|Money|Environment) (Show|Hide) subsections", "", t)
        t = re.sub("WAtoday", "", t)
        t = re.sub("The Sydney Morning Herald", "", t)
        t = re.sub("(Exclusive|Opinion) Normal text", "", t)
        t = re.sub("Normal text", "", t)
        t = re.sub("sizeLarger", "", t)
        t = re.sub("text sizeVery", "", t)
        t = re.sub("large text size", "", t)
        t = re.sub("Skip to sections navigation", "", t)
        t = re.sub("Skip to content", "", t)
        t = re.sub("Skip to footer", "", t)
        t = re.sub("Open Menu", "", t)
        t = re.sub("Replay Video", "", t)
        t = re.sub("Play Video", "", t)
        t = re.sub("Playing in 5 ... Don't Play", "", t)
        t = re.sub("Playing in  ... Don't Play", "", t)
        t = re.sub("Latest in Video", "", t) 
        t = re.sub("\nLoading \n", "", t)
        t = re.sub("Most Popular  \nLoading", "", t)
        t = re.sub("http:\/\/[\w\.]+", "", t) #remove web addresses
        t = re.sub("\/\/[\w\.]+", "", t) #remove web addresses
        t = re.sub("[\w.]*@[\w.]+", "", t) #remove emails or twitter accounts
        t = re.sub("Mr |Ms |Mrs ", "", t)
        t = re.sub("(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*",
                   "",t)
        
        t1 = t.split()
        t = ' '.join(filter(lambda x: x not in names, [y for y in t1])) #remove names
        t = re.sub("Login", "", t)
        t = re.sub("normal text", "", t)
        t = re.sub("sizeLarger", "", t)
        t = re.sub("text", "", t)
        t = re.sub("sizeVery", "", t)
        t = re.sub("large", "", t)
        t = re.sub("size", "", t)
        t = re.sub("normal", "", t)
        t = re.sub("yes", "", t)
        t = re.sub("subsections", "", t)
        t = re.sub("[^a-zA-Z\- ]", "", t)
        t = re.sub("javascript", "", t)
        t = re.sub("http", "", t)
        t = re.sub("www", "", t)
        clean.append(t)
    clean_text.extend(clean)
    return clean_text


def deleteShortContent(content):
    new_content = []
    for text in content:
        if len(text) > 300:
            new_content.append(text)
        #else:
            #print("======Deleted short text=======")
            #print(text)
    return new_content


def vectorise(data):
    tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    tfidf_matrix = tfidf.fit_transform(data)
    input_data = tfidf_matrix.toarray()
    features = tfidf.get_feature_names()
    return input_data, features


def save_data(data):
    with h5py.File('input_data3.h5', 'w') as hf:
        hf.create_dataset("input_data3",  data=data)


def save_features(features):
    with open('feature3.txt', 'w') as file:  
        for item in features:
            file.write('%s\n' % item)

def save_text(text):
    with open('clean_text.txt', 'w') as file:  
        for item in text:

            file.write('%s\n' % item)

#download_nltk()

from loaddata_raw import load_data

dir='/Users/preronamajumder/Documents/COMP5703/data/' #data fold path
args = 'content'  #data type
max_words = 2000  #dimensons 
round = 3   #select which hearing

data = load_data(dir,round,args)
print(len(data))
#print(data[0:2])
data = clean_text(data)                     #remove unnecesary words
#print(data)
data = deleteShortContent(data)
print(data)
print(len(data))
save_text(data)
token_list, word_list = get_tokens(data)    #tokenise. get PoST tags for each word
#print(len(word_list)) 
phrase_list = get_phrases(token_list)       #get phrases defined by grammatical rules
#word_list = stemming(word_list)
#print(phrase_list)
#print(len(phrase_list))
#print(len(word_list[0]))
print("Vectorising data...")
phrase_data, phrase_features = vectorise(phrase_list)   #get tfidf values with phrases as features
#word_data, word_features = vectorise(word_list)
print(phrase_data.shape)
#print(phrase_features[0:200])

#print(word_data.shape)
#print(word_features)
print("Saving data...")
#save_data(phrase_data)                      #save feature matrix (phrases)
print("Saving features...")
#save_features(phrase_features)              #save feature (phrase) list
