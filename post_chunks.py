import nltk
import h5py
from sklearn.feature_extraction.text import TfidfVectorizer
import ssl

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
    token_list=[]
    for d in data:
        token = nltk.word_tokenize(d)
        x = nltk.pos_tag(token)       
        token_list.append(x)
    return token_list
    

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


#download_nltk()

from loaddata_raw import load_data

dir='/Users/preronamajumder/Documents/COMP5703/data/' #data fold path
args = 'content'  #data type
max_words = 2000  #dimensons 
round = 5   #select which hearing

data = load_data(dir,round,args)
print(len(data))

#token = nltk.word_tokenize(data[1])

#x = nltk.pos_tag(token)

token_list = get_tokens(data)
phrase_list = get_phrases(token_list)

tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
tfidf_matrix = tfidf.fit_transform(phrase_list)
input_data = tfidf_matrix.toarray()
print(input_data.shape)
#print(tfidf.get_feature_names())

with h5py.File('input_data5.h5', 'w') as hf:
    hf.create_dataset("input_data5",  data=input_data)
