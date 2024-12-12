import re
import sys
import os
import numpy as np
import contractions
import nltk
import glob

import numpy as np
import sklearn
import nltk
import gensim

from gensim.models import KeyedVectors

from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


get_ipython().system('wget -N http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
get_ipython().system('tar -xvzf aclImdb_v1.tar.gz')
get_ipython().system("rm aclImdb_v1.tar.gz")
PATH_TO_GUTENBERG_CORPUS = sys.argv[1] #Set Path 
SCRIPT_DIRECTORY = sys.argv[2] #Set Script Directory

def download_corpus(corpus="gutenberg"):
    """Download Project Gutenberg corpus, consisting of 18 classic books
    Book list:
       ['austen-emma.txt',
        'austen-persuasion.txt',
        'austen-sense.txt',
        'bible-kjv.txt',
        'blake-poems.txt',
        'bryant-stories.txt',
        'burgess-busterbrown.txt',
        'carroll-alice.txt',
        'chesterton-ball.txt',
        'chesterton-brown.txt',
        'chesterton-thursday.txt',
        'edgeworth-parents.txt',
        'melville-moby_dick.txt',
        'milton-paradise.txt',
        'shakespeare-caesar.txt',
        'shakespeare-hamlet.txt',
        'shakespeare-macbeth.txt',
        'whitman-leaves.txt']
    """
    nltk.download(corpus)
    raw = nltk.corpus.__getattr__(corpus).raw()

    return raw


def identity_preprocess(s):
    return s


def clean_text(s):
    s = s.strip()  # strip leading / trailing spaces
    s = s.lower()  # convert to lowercase
    s = contractions.fix(s)  # e.g. don't -> do not, you're -> you are
    s = re.sub("\s+", " ", s)  # strip multiple whitespace
    s = re.sub(r"[^a-z\s]", " ", s)  # keep only lowercase letters and spaces

    return s


def tokenize(s):
    tokenized = [w for w in s.split(" ") if len(w) > 0]  # Ignore empty string

    return tokenized


def preprocess(s):
    return tokenize(clean_text(s))


def process_file(corpus, preprocess=identity_preprocess):
    lines = [preprocess(ln) for ln in corpus.split("\n")]
    lines = [ln for ln in lines if len(ln) > 0]  # Ignore empty lines

    return lines


def predict_word(word,mode=0,transducer = 'L',verbose=0):
    #Write FST acceptor txt of word to be corrected
    with open('./vocab/fsts/wrong_word.txt','w') as f:
        counter = 0 
        start = True
        for char in word:
            if(start):
                f.write(("{} {} {} {} \n").format(str(0),str(counter+1),char,0))
                start = False
                counter+=1
                continue
            else:
                f.write(("{} {} {} {} \n").format(str(counter),str(counter+1),char,0))
                counter+=1
        f.write(("{}\n").format(str(counter)))
    #Choose Required Orthograph for task and number of words to be predicted
    if(mode==0):
        if(transducer=='L'):
            get_ipython().system(" ./scripts/cpredict.sh 5 'L'")
        elif(transducer=='E'):
            get_ipython().system(" ./scripts/cpredict.sh 5 'E'")
        elif(transducer=='LW'):
            get_ipython().system(" ./scripts/cpredict.sh 5 'LW'")
        elif(transducer=='EW'):
            get_ipython().system(" ./scripts/cpredict.sh 5 'EW'")
    else: 
        if(transducer=='L'):
            get_ipython().system(" ./scripts/cpredict.sh 1 'L'")
        elif(transducer=='E'):
            get_ipython().system(" ./scripts/cpredict.sh 1 'E'")
        elif(transducer=='LW'):
            get_ipython().system(" ./scripts/cpredict.sh 1 'LW'")
        elif(transducer=='EW'):
            get_ipython().system(" ./scripts/cpredict.sh 1 'EW'")
    #Extract predicted words from fstprint of composed spell checker
    possible_words = []
    with open('./vocab/fsts/checked_print.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub(r"[^a-zA-Z]+", ' ', line)
            line = line.strip()
            line = line.replace("epsilon", "")
            if(len(line)!=0):
                line = line.split(' ')[1]
                if(len(line) > 1):
                    possible_words.append(line)
    #Print results if desirable
    if(verbose==1):
        print("Wrong word is:",word)

        print("Possible words are: ",end="")
        temp = possible_words.copy()
        for word in possible_words:
            if(len(temp)!=1):
                print(temp[0]+',',end=" ")
                temp.pop(0)
            else:
                print(temp[0]+'.',end="\n\n")
                
    return possible_words



def read_syms(filename):
    #Read contents of syms file
    f = open('./vocab/{}.syms'.format(filename),'r')
    lines = f.readlines()
    arr = []
    for x in lines:
        arr.append(x.split('\t')[0])
    f.close()
    return arr


def make_fst(filename,word):
    with open('./vocab/fsts/{}.txt'.format(filename),'w') as f:
        counter = 0 
        start = True
        for char in word:
            if(start):
                f.write(("{} {} {} {} \n").format(str(0),str(counter+1),char,0))
                start = False
                counter+=1
                continue
            else:
                f.write(("{} {} {} {} \n").format(str(counter),str(counter+1),char,0))
                counter+=1
        f.write(("{}\n").format(str(counter)))


# ## Step 1: Downloading Gutenberg Corpus
print("----------STEP 1----------\n\n")
#PATH_TO_GUTENBERG_CORPUS = '/home/nick/nltk_data/corpora/gutenberg' #Set Path for Gutenberg File
CORPUS = "gutenberg"
raw_corpus = download_corpus(corpus=CORPUS)
preprocessed = process_file(raw_corpus, preprocess=preprocess)

for words in preprocessed:
    sys.stdout.write(" ".join(words))
    sys.stdout.write("\n")

print(sys.argv[0])
txts = os.listdir(PATH_TO_GUTENBERG_CORPUS)
txts


# ## Step 2: Creating Dictionaries from every .txt file


data_txt = {}
for txt in txts:
    if(txt == 'README'):
        continue 
    with open(PATH_TO_GUTENBERG_CORPUS+'/{}'.format(txt), 'r', encoding='latin-1') as f: 
        # Read the contents of the file into a list 
        print(txt)
        lines = f.readlines() 
        # Create an empty dictionary 
        # Loop through the list of lines 
        for line in lines: 
            line = re.sub(r"[^a-zA-Z ]+", '', line)
            # Split the line into key-value pairs 
            keys = line.split(' ')
            for key in keys:
                if(key.lower() in data_txt.keys()):    
                    data_txt[key.lower()] += 1
                else:
                    data_txt[key.lower()] = 1
                    
#Remove character '' and tokens that have appeared less than 5 times                
keys = data_txt.keys()
keys_to_remove = ['']
for key in keys:
    if(data_txt[key] < 5):
        keys_to_remove.append(key)

for key in keys_to_remove:
    data_txt.pop(key, None)

#Write words in txt file
with open('./vocab/vocab.txt','w') as f:
    for key in data_txt.keys():
        f.write((key+'\t'+str(data_txt[key])+'\n').lower())


# ## Step 3: Creating character symbols and word symbols


import string
#Create syms file for all lowercase english characters and words in vocabulary
alphabet = list(string.ascii_lowercase)

counter = 1

ascii_syms = {'<epsilon>':0}
for letter in alphabet:
    ascii_syms[letter] = counter
    counter += 1  
    
with open('./vocab/chars.syms','w') as f:
    for key in ascii_syms.keys():
        f.write(key+'\t'+str(ascii_syms[key])+'\n')




f = open('./vocab/vocab.txt','r')
lines = f.readlines()
words = []
for x in lines:
    words.append(x.split('\t')[0])
f.close()

counter = 1

tokens = {'<epsilon>':0}
for word in words:
    tokens[word] = counter
    counter += 1  
    
with open('./vocab/words.syms','w') as f:
    for key in tokens.keys():
        f.write(key+'\t'+str(tokens[key])+'\n')



chars = read_syms('chars')


# ## Step 4: Creating .txt files for compiling fst



#One State Transducer
with open('./vocab/fsts/L.txt','w') as f:  
    #Deletion - Weight 1
    for char in chars:
        if(char == '<epsilon>'):
            continue
        
        f.write(("0 0 {} <epsilon> {}\n").format(char,1))
        
    #Insertion - Weight 1
    for char in chars:
        if(char == '<epsilon>'):
            continue
        f.write(("0 0 <epsilon> {} {}\n").format(char,1))
                
    #Pass to other character or same character - Weight 1, 0 respectively
    for char1 in chars:
        for char2 in chars:
            if(char1 == '<epsilon>' or char2 == '<epsilon>'):
                continue
            if(char1 == char2):
                f.write(("0 0 {} {} {}\n").format(char1, char2, 0))
                continue
            f.write("0 0 {} {} {}\n".format(char1, char2, 1))
            
    #Write End State 0
    f.write("0 0")




#Create Low Character Transducer for easier representatioon
with open('./vocab/fsts/L_low.txt','w') as f:
    stop_char = 'd'
    for char in chars:
        if(char == '<epsilon>'):
            continue
        if(char == stop_char):
            break
        f.write(("0 0 {} <epsilon> {}\n").format(char,1))
                
    for char in chars:
        if(char == '<epsilon>'):
            continue
        if(char == stop_char):
            break
        f.write(("0 0 <epsilon> {} {}\n").format(char,1))
                
    low_data = False
    for char1 in chars:
        if(low_data):
            break
        for char2 in chars:
            if(char1 == '<epsilon>' or char2 == '<epsilon>'):
                continue
            if(char1 == 'b'):
                low_data = True
                break
            if(char1 == char2):
                f.write(("0 0 {} {} {}\n").format(char1, char2, 0))
                continue
            f.write("0 0 {} {} {}\n".format(char1, char2, 1))
    f.write("0 0")




#Compile full and low character transducers
get_ipython().system(' fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms vocab/fsts/L.txt vocab/fsts/L.bin.fst')

get_ipython().system(' fstprint -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms vocab/fsts/L.bin.fst > vocab/fsts/L_print.txt')

get_ipython().system(' fstdraw -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms vocab/fsts/L.bin.fst | dot -Tpng  > vocab/fsts/L.png')




get_ipython().system(' fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms vocab/fsts/L_low.txt vocab/fsts/L_low.bin.fst')

get_ipython().system(' fstprint -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms vocab/fsts/L_low.bin.fst > vocab/fsts/L_low_print.txt')

get_ipython().system(' fstdraw -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms vocab/fsts/L_low.bin.fst | dot -Tpng  > vocab/fsts/L_low.png')


# ## Step 5: Lexicon Acceptor


#Create Word Lexicon from syms file
words = read_syms('words')


with open('./vocab/fsts/V.txt','w') as f:
    counter = 0
    final_states = []
    for word in words:
        if(word == '<epsilon>'):
            continue
        start = True
        last_char = 0
        #For every character in syms, we create a tranducer that accepts that word
        #We then add every single word transducer into one using the .txt file, without union function
        for char in word:
            if(last_char == len(word)-1):
                final_states.append(counter+1)
            if(start):
                #If the input is the word, we pass through all its characters with zero cost with no input
                f.write(("{} {} {} {} {}\n").format(0,str(counter+1),char,word,0))
                start = False
                counter+=1
                last_char+=1
                continue
            else:
                f.write(("{} {} {} {} {}\n").format(str(counter),str(counter+1),char,'<epsilon>',0))
                counter+=1
                last_char+=1

    #Write end states
    for state in final_states:
        if(state != final_states[-1]):
            f.write(("{}\n").format(str(state)))
        else:
            f.write(("{}").format(str(state)))




#Create Low-Word Lexicon Acceptor for easier representation
with open('./vocab/fsts/V_low.txt','w') as f:
    counter = 0
    word_counter = 0
    final_states = []
    for word in words:
        if(word_counter == 4):
            break
        if(word == '<epsilon>'):
            continue
        start = True
        last_char = 0
        for char in word:
            if(last_char == len(word)-1):
                final_states.append(counter+1)
            if(start):
                f.write(("{} {} {} {} {}\n").format(str(0),str(counter+1),char,word,0))
                first_automata = False
                start = False
                counter+=1
                last_char+=1
                continue
            else:
                f.write(("{} {} {} {} {}\n").format(str(counter),str(counter+1),char,'<epsilon>',0))
                counter+=1
                last_char+=1
        word_counter+=1 
    for state in final_states:
        f.write(("{}\n").format(str(state)))




#Run .sh file to create V acceptor and V_low acceptor, as well as images and print files
get_ipython().system(' ./scripts/orthograph.sh')


# ## Step 6: Compose L and V transducer and acceptor (Orthograph)



#Combine both Levenshtein trasducer and our lexicon acceptor 
#Sort the arcs of graphs on output and input for L and V, respectively
get_ipython().system(' fstarcsort --sort_type=olabel vocab/fsts/L.bin.fst vocab/fsts/L_sorted.bin.fst')
get_ipython().system(' fstarcsort --sort_type=ilabel vocab/fsts/V_opt.bin.fst vocab/fsts/V_sorted.bin.fst')
get_ipython().system(' fstcompose vocab/fsts/L_sorted.bin.fst vocab/fsts/V_sorted.bin.fst vocab/fsts/LV.bin.fst')




get_ipython().system(' fstprint -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/LV.bin.fst > vocab/fsts/LV_print.txt')


# ## Step 7: Check Results of Orthograph


print("----------STEP 7----------\n\n")
word = 'cwt'
with open('./vocab/fsts/wrong_word.txt','w') as f:
    counter = 0 
    start = True
    for char in word:
        if(start):
            f.write(("{} {} {} {} \n").format(str(0),str(counter+1),char,0))
            first_automata = False
            start = False
            counter+=1
            continue
        else:
            f.write(("{} {} {} {} \n").format(str(counter),str(counter+1),char,0))
            counter+=1
    f.write(("{}\n").format(str(counter)))

predict_word(word,transducer = 'L',verbose = 1,mode=0)


word = 'cit'
with open('./vocab/fsts/wrong_word.txt','w') as f:
    counter = 0 
    start = True
    for char in word:
        if(start):
            f.write(("{} {} {} {} \n").format(str(0),str(counter+1),char,0))
            first_automata = False
            start = False
            counter+=1
            continue
        else:
            f.write(("{} {} {} {} \n").format(str(counter),str(counter+1),char,0))
            counter+=1
    f.write(("{}\n").format(str(counter)))

predict_word(word,transducer = 'L',verbose = 1,mode=0)


def predict_txt(path,transducer = 'L',verbose = 1,mode = 0,full=False):
    #Choose whether to predict entire spell_test.txt or 20 first words
    if(not full):
        with open(path,'r') as f:
            head = [next(f).replace('\n',"").split(':')[1].strip().split(' ') for _ in range(20)]
        with open(path,'r') as f:    
            correct = [next(f).replace('\n',"").split(':')[0].strip() for _ in range(20)]
    else:
        with open(path,'r') as f:
            head = [next(f).replace('\n',"").split(':')[1].strip().split(' ') for _ in range(141)]
        with open(path,'r') as f:    
            correct = [next(f).replace('\n',"").split(':')[0].strip() for _ in range(141)]
    accuracy = []
    index = 0
    hits = 0
    length = 0
    #Run predict word with assigned parameters for every word
    with open('./vocab/{}_predictions.txt'.format(transducer),'w') as f:
        for words in head:
            for word in words:
                arr = predict_word(word,transducer=transducer,verbose=verbose,mode=mode)
                f.write("{} : {}\n".format(word,arr))
                #Calculate how many hits the spell checker had
                for corrected in arr:
                    length+=1
                    if(corrected == correct[index]):
                        hits+=1
            index+=1
        #Print total accuracy at the end of the predictions
        if(mode == 1):
        	print("Total accuracy for transducer {} for top word is: {:.2f}%\n".format(transducer,hits*100/length))
        elif(mode == 0):
        	print("Total accuracy for transducer {} for top 5 words is: {:.2f}%\n".format(transducer,hits*100/length))


#Run for first 5 words and for the top word and compare accuracies
predict_txt('./data/spell_test.txt',transducer='L',verbose=0,mode=0)
predict_txt('./data/spell_test.txt',transducer='L',verbose=0,mode=1)


# ## Step 8: Calculating Cost of Edits - Updated Levenshtein Transducer E

print("----------STEP 8----------\n\n")
#Extract information for characters changes from wiki txt
with open('./data/wiki.txt','r') as f:
    c_words = []
    w_words = []
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.replace('\n','')
        split = line.split('\t')
        w_words.append(split[0])
        c_words.append(split[1])
        
c_word = c_words[0]
w_word = w_words[0]
print(c_word,w_word)


#Make txt files to compile for wrong and correct word
make_fst('w_word',w_word)
make_fst('c_word',c_word)


get_ipython().system(' fstcompile -isymbols=vocab/chars.syms --acceptor=true vocab/fsts/w_word.txt vocab/fsts/w_word.bin.fst')

get_ipython().system(' fstcompose vocab/fsts/w_word.bin.fst vocab/fsts/L.bin.fst vocab/fsts/ML.bin.fst')


#Shortesh path returns the required edits,ie Levenshtein distance and what type of edits
get_ipython().system(' fstcompile -isymbols=vocab/chars.syms --acceptor=true vocab/fsts/c_word.txt vocab/fsts/c_word.bin.fst')

get_ipython().system(' fstcompose vocab/fsts/ML.bin.fst vocab/fsts/c_word.bin.fst vocab/fsts/MLN.bin.fst')

get_ipython().system(' fstshortestpath vocab/fsts/MLN.bin.fst | fstprint --isymbols=vocab/chars.syms --osymbols=vocab/chars.syms --show_weight_one | grep -v "0$"| cut -d$\'\\t\' -f3-4')


from os.path import exists
#Check if file exists. If it does, empty it. If it doesn't, create it.
if(exists('./vocab/fsts/changes.txt')):
    get_ipython().system(' > vocab/fsts/changes.txt')
else:
    get_ipython().system(' touch vocab/fsts/changes.txt')


#Run shortest path for every word and write results in changes.txt file
#Special characters were chosen that differed from original chars.syms were chosen to be ignored
for i in range(len(c_words)):
    c_word = c_words[i]
    w_word = w_words[i]
    make_fst('w_word',w_word)
    make_fst('c_word',c_word)
    
    get_ipython().system(' fstcompile -isymbols=vocab/chars.syms --acceptor=true vocab/fsts/w_word.txt vocab/fsts/w_word.bin.fst')

    get_ipython().system(' fstcompose vocab/fsts/w_word.bin.fst vocab/fsts/L.bin.fst vocab/fsts/ML.bin.fst')
    
    get_ipython().system(' fstcompile -isymbols=vocab/chars.syms --acceptor=true vocab/fsts/c_word.txt vocab/fsts/c_word.bin.fst')
    
    get_ipython().system(' fstcompose vocab/fsts/ML.bin.fst vocab/fsts/c_word.bin.fst vocab/fsts/MLN.bin.fst')
    
    get_ipython().system(' fstshortestpath vocab/fsts/MLN.bin.fst | fstprint --isymbols=vocab/chars.syms --osymbols=vocab/chars.syms --show_weight_one | grep -v "0$"| cut -d$\'\\t\' -f3-4 >> vocab/fsts/changes.txt')


#Extract information and frequencies from created changes.txt file
with open('./vocab/fsts/changes.txt','r') as f:
    freqs = {}
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n','')
        key = tuple(line.split('\t'))
        if(key in freqs.keys()):
            freqs[key] += 1
        else:
            freqs[key] = 1
    sum_of_changes = sum(freqs.values())
    freqs.update( (k,-np.log10(freqs[k]/sum_of_changes)) for k in freqs)


#Recreate Levenshtein transducer with updated weights for each edit
#One State Transducer
with open('./vocab/fsts/E.txt','w') as f:  
    #Deletion - Weight 1
    for char in chars:
        if(char == '<epsilon>'):
            continue
        try:
            #If edit exists, set weight as calculated
            f.write(("0 0 {} <epsilon> {}\n").format(char,freqs[(char,'<epsilon>')]))
        except:
            #If it doesn't, set value to 10^6
            f.write(("0 0 {} <epsilon> {}\n").format(char,1e6))
        
    #Insertion - Weight 1
    for char in chars:
        if(char == '<epsilon>'):
            continue
        try:
            f.write(("0 0 <epsilon> {} {}\n").format(char,freqs[('<epsilon>',char)]))
        except:
            f.write(("0 0 <epsilon> {} {}\n").format(char,1e6))
                
    #Pass to other character or same character - Weight 1, 0 respectively
    for char1 in chars:
        for char2 in chars:
            if(char1 == '<epsilon>' or char2 == '<epsilon>'):
                continue
            if(char1 == char2):
                f.write(("0 0 {} {} {}\n").format(char1, char2, 0))
                continue
            try:
                f.write(("0 0 {} {} {}\n").format(char1,char2,freqs[(char1,char2)]))
            except:
                f.write(("0 0 {} {} {}\n").format(char1,char2,1e6))
            
    #Write End State 0
    f.write("0 0")


#Compile Transducer
get_ipython().system(' fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms vocab/fsts/E.txt vocab/fsts/E.bin.fst')

get_ipython().system(' fstprint -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms vocab/fsts/E.bin.fst > vocab/fsts/E_print.txt')

get_ipython().system(' fstdraw -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms vocab/fsts/E.bin.fst | dot -Tpng  > vocab/fsts/E.png')


#Combine both Updated Levenshtein transducer and our lexicon acceptor 
get_ipython().system(' fstarcsort --sort_type=olabel vocab/fsts/E.bin.fst vocab/fsts/E_sorted.bin.fst')
get_ipython().system(' fstarcsort --sort_type=ilabel vocab/fsts/V_opt.bin.fst vocab/fsts/V_sorted.bin.fst')
get_ipython().system(' fstcompose vocab/fsts/E_sorted.bin.fst vocab/fsts/V_sorted.bin.fst vocab/fsts/EV.bin.fst')


word = 'cit'
with open('./vocab/fsts/wrong_word.txt','w') as f:
    counter = 0 
    start = True
    for char in word:
        if(start):
            f.write(("{} {} {} {} \n").format(str(0),str(counter+1),char,0))
            first_automata = False
            start = False
            counter+=1
            continue
        else:
            f.write(("{} {} {} {} \n").format(str(counter),str(counter+1),char,0))
            counter+=1
    f.write(("{}\n").format(str(counter)))

predict_word(word,transducer = 'E',verbose = 1,mode=0)


word = 'cwt'
with open('./vocab/fsts/wrong_word.txt','w') as f:
    counter = 0 
    start = True
    for char in word:
        if(start):
            f.write(("{} {} {} {} \n").format(str(0),str(counter+1),char,0))
            first_automata = False
            start = False
            counter+=1
            continue
        else:
            f.write(("{} {} {} {} \n").format(str(counter),str(counter+1),char,0))
            counter+=1
    f.write(("{}\n").format(str(counter)))

predict_word(word,transducer = 'E',verbose = 1,mode=0)


#Run for first 5 words and for the top word and compare accuracies
predict_txt('./data/spell_test.txt',transducer='E',verbose=0,mode=0)
predict_txt('./data/spell_test.txt',transducer='E',verbose=0,mode=1)


#Making Dictionary for frequency of words
word_freqs = data_txt.copy()
sum_of_changes = sum(data_txt.values())
word_freqs.update( (k,-np.log10(word_freqs[k]/sum_of_changes)) for k in word_freqs)


#Create word acceptor with weight set as weight calculated from vocabulary
with open('./vocab/fsts/W.txt','w') as f:
    for word in words:
        if(word == '<epsilon>'):
            continue
        f.write(("0 0 {} {}\n").format(word,word_freqs[word])) 

    f.write('0 0')


with open('./vocab/fsts/W_low.txt','w') as f:
    counter = 0 
    for word in words:
        if(word == '<epsilon>'):
            continue
        if(counter == 4):
            break
        f.write(("0 0 {} {}\n").format(word,word_freqs[word])) 
        counter+=1

    f.write('0 0')


#Compile W acceptor
get_ipython().system(' fstcompile -isymbols=vocab/words.syms --acceptor=true vocab/fsts/W.txt vocab/fsts/W.bin.fst')

get_ipython().system(' fstrmepsilon vocab/fsts/W.bin.fst > vocab/fsts/W_opt_1.bin.fst')

get_ipython().system(' fstdeterminize vocab/fsts/W_opt_1.bin.fst > vocab/fsts/W_opt_2.bin.fst')

get_ipython().system(' fstminimize vocab/fsts/W_opt_2.bin.fst > vocab/fsts/W_opt.bin.fst')

get_ipython().system(' fstprint -isymbols=vocab/words.syms -osymbols=vocab/words.syms vocab/fsts/W_opt.bin.fst > vocab/fsts/W_print.txt')


#Compile W acceptor with low data
get_ipython().system(' fstcompile -isymbols=vocab/words.syms --acceptor=true vocab/fsts/W_low.txt vocab/fsts/W_low.bin.fst')

get_ipython().system(' fstrmepsilon vocab/fsts/W_low.bin.fst > vocab/fsts/W_low_opt_1.bin.fst')

get_ipython().system(' fstdeterminize vocab/fsts/W_low_opt_1.bin.fst > vocab/fsts/W_low_opt_2.bin.fst')

get_ipython().system(' fstminimize vocab/fsts/W_low_opt_2.bin.fst > vocab/fsts/W_low_opt.bin.fst')

get_ipython().system(' fstdraw -isymbols=vocab/words.syms -osymbols=vocab/words.syms vocab/fsts/W_low_opt.bin.fst | dot -Tpng  > vocab/fsts/W_low.png')


#Compose VW_low
get_ipython().system(' fstcompose vocab/fsts/V_low_opt.bin.fst vocab/fsts/W_low_opt.bin.fst vocab/fsts/VW_low.bin.fst')

get_ipython().system(' fstdraw -isymbols=vocab/words.syms -osymbols=vocab/words.syms vocab/fsts/VW_low.bin.fst | dot -Tpng  > vocab/fsts/VW_low.png')


#Creating orthograph with Vanilla-Levenshtein and updated word frequencies
get_ipython().system(' fstarcsort --sort_type=olabel vocab/fsts/LV.bin.fst vocab/fsts/LV_sorted.bin.fst')

get_ipython().system(' fstarcsort --sort_type=ilabel vocab/fsts/W_opt.bin.fst vocab/fsts/W_sorted.bin.fst')

get_ipython().system(' fstcompose vocab/fsts/LV_sorted.bin.fst vocab/fsts/W_sorted.bin.fst vocab/fsts/LVW.bin.fst')

get_ipython().system(' fstprint -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/LVW.bin.fst > vocab/fsts/LVW_print.txt')


#Creating orthograph with Updated-Levenshtein and updated word frequencies
get_ipython().system(' fstarcsort --sort_type=olabel vocab/fsts/EV.bin.fst vocab/fsts/EV_sorted.bin.fst')

get_ipython().system(' fstcompose vocab/fsts/EV_sorted.bin.fst vocab/fsts/W_sorted.bin.fst vocab/fsts/EVW.bin.fst')

get_ipython().system(' fstprint -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/EVW.bin.fst > vocab/fsts/EVW_print.txt')


word = 'cit'
with open('./vocab/fsts/wrong_word.txt','w') as f:
    counter = 0 
    start = True
    for char in word:
        if(start):
            f.write(("{} {} {} {} \n").format(str(0),str(counter+1),char,0))
            first_automata = False
            start = False
            counter+=1
            continue
        else:
            f.write(("{} {} {} {} \n").format(str(counter),str(counter+1),char,0))
            counter+=1
    f.write(("{}\n").format(str(counter)))

predict_word(word,transducer = 'LW',verbose = 1,mode=0)


word = 'cwt'
with open('./vocab/fsts/wrong_word.txt','w') as f:
    counter = 0 
    start = True
    for char in word:
        if(start):
            f.write(("{} {} {} {} \n").format(str(0),str(counter+1),char,0))
            first_automata = False
            start = False
            counter+=1
            continue
        else:
            f.write(("{} {} {} {} \n").format(str(counter),str(counter+1),char,0))
            counter+=1
    f.write(("{}\n").format(str(counter)))

predict_word(word,transducer = 'LW',verbose = 1,mode=0)


#Run for first 5 words and for the top word and compare accuracies
predict_txt('./data/spell_test.txt',transducer='LW',verbose=0,mode=0)
predict_txt('./data/spell_test.txt',transducer='LW',verbose=0,mode=1)


#Run for first 5 words and for the top word and compare accuracies
predict_txt('./data/spell_test.txt',transducer='EW',verbose=0,mode=0)
predict_txt('./data/spell_test.txt',transducer='EW',verbose=0,mode=1)


# ## Βήμα 10 - Έλεγχος Ορθογράφων στο test set

#Check accuracies for top word in entirety of spell_test.txt file
print('------------Calculating Accuracies on entire spell_test.txt------------')
#predict_txt('./data/spell_test.txt',transducer='L',verbose=0,mode=1,full=True)
#predict_txt('./data/spell_test.txt',transducer='E',verbose=0,mode=1,full=True)
#predict_txt('./data/spell_test.txt',transducer='LW',verbose=0,mode=1,full=True)
#predict_txt('./data/spell_test.txt',transducer='EW',verbose=0,mode=1,full=True)


# # Part 2

# ## Βήμα 12: Εξαγωγή αναπαραστάσεων word2vec

print("----------STEP 12----------\n\n")
txts = os.listdir(PATH_TO_GUTENBERG_CORPUS)

data_txt = []

for txt in txts:
    if(txt == 'README'):
        continue 
    with open(PATH_TO_GUTENBERG_CORPUS + '/{}'.format(txt), 'r', encoding='latin-1') as f: 
        # Read the contents of the file into a list 
        data = f.read()
        tokens = sent_tokenize(data)
        for i in range(len(tokens)):
            tokens[i] = re.sub(r'[^a-zA-Z ]+', ' ', tokens[i]).strip()
            tokens[i] = [x.lower() for x in tokens[i].split(" ") if x != '']
            data_txt.append(tokens[i])




from gensim.models import Word2Vec

#Train word2vec model for 100 dimensions, window equal to 5 and 1000 epochs
print("----------Training Model----------")
model = Word2Vec(sentences = data_txt, vector_size=100, window=5, min_count=1, workers=4,epochs=1000)

#Save KeyedVectors for future use
model.save("word2vec.model")


words = ['bible', 'book', 'bank', 'water']


# > Results for vectors from Gutenberg Corpus

#Check highest similarity for words in array "words"
for word in words:
    print("Most related words to '{}' are:".format(word))
    arr = model.wv.most_similar(positive = [word], topn = 5)
    for i in range(len(arr)):
        print("{} with cos similarity {:.3f}".format(arr[i][0], arr[i][1]))
    print('\n')


#Create word triples for subtraction-addition experimenting
triplets = [['girls', "queen", "king"], ['taller', 'tall', 'good'], ['france', 'paris', 'london']]


for triplet in triplets:
    v = model.wv[triplet[0]] - model.wv[triplet[1]] + model.wv[triplet[2]]
    arr = model.wv.most_similar(positive = v, topn = 3)
    print("Most related words to analogy vector are:")
    for i in range(len(arr)):
        print("{} with cos similarity {:.3f}".format(arr[i][0], arr[i][1]))
    print('\n')


# > Results for GoogleNews vectors

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True, limit=100000)


for word in words:
    print("Most related words to '{}' are:".format(word))
    arr = model.most_similar(positive = [word], topn = 5)
    for i in range(len(arr)):
        print("{} with cos similarity {:.3f}".format(arr[i][0], arr[i][1]))
    print('\n')


triplets = [['girls', "queen", "king"], ['taller', 'tall', 'good'], ['France', 'Paris', 'London']]


for triplet in triplets:
    v = model[triplet[0]] - model[triplet[1]] + model[triplet[2]]
    arr = model.most_similar(positive = v, topn = 3)
    print("Most related words to analogy vector are:")
    for i in range(len(arr)):
        print("{} with cos similarity {:.3f}".format(arr[i][0], arr[i][1]))
    print('\n')


# ## Βήμα 13: Οπτικοποίηση των word embeddings
print("----------STEP 13----------\n\n")
# > Results for vectors from Gutenberg Corpus
print("Creating Embeddings and Metadata .tsvs
#Reload KeyedVectors if required
model = Word2Vec.load("word2vec.model")

#Create .tsvs for u se in online projection tool
counter = 0
with open('./embeddings.tsv', 'w') as f:
    for vector in model.wv.vectors:
        counter += 1
        for i in range(vector.shape[0]):
            if (i == 99):
                if (counter == model.wv.vectors.shape[0]):
                    f.write('{}'.format(vector[i]))
                    break
                f.write('{}\n'.format(vector[i]))
                break
            f.write('{}\t'.format(vector[i]))


counter = 0
with open('./metadata.tsv', 'w') as f:
    for word in model.wv.index_to_key:
        counter += 1
        if (counter == len(model.wv.index_to_key)):
            f.write('{}'.format(word))
            break
        f.write('{}\n'.format(word))


# ## Βήμα 14: Ανάλυση συναισθήματος με word2vec embeddings

print("----------STEP 14----------\n")


data_dir = os.path.join(SCRIPT_DIRECTORY, "aclImdb/")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
pos_train_dir = os.path.join(train_dir, "pos")
neg_train_dir = os.path.join(train_dir, "neg")
pos_test_dir = os.path.join(test_dir, "pos")
neg_test_dir = os.path.join(test_dir, "neg")

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 5000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000


SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(SEED)


def strip_punctuation(s):
    return re.sub(r"[^a-zA-Z\s]", " ", s)


def preprocess(s):
    return re.sub("\s+", " ", strip_punctuation(s).lower())


def tokenize(s):
    return s.split(" ")


def preproc_tok(s):
    return tokenize(preprocess(s))


def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, "*.txt"))
    data = []

    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        with open(sample, "r") as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)

    return data


def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)

    return list(corpus[indices]), list(y[indices])


def extract_nbow(corpus,model,google,length):
    """Extract neural bag of words representations"""
    nbow = np.zeros(length)
    for sentence in corpus:
        tokenized_sentence = preproc_tok(sentence.strip())
        nbag = np.zeros(length)
        for word in tokenized_sentence:
            if(not google):
                #Check if word exists in embeddings or not
                try:
                    nbag = nbag + model.wv[word]
                except:
                    nbag = nbag + np.zeros(length)
            else:
                try:
                    nbag = nbag + model[word]
                except:
                    nbag = nbag + np.zeros(length)
        nbag = nbag / len(tokenized_sentence)
        nbow = np.vstack((nbow, nbag))
    #Delete first row that was used for initialization    
    nbow = np.delete(nbow, obj = 1, axis = 0)
    return nbow


def train_sentiment_analysis(train_corpus, train_labels):
    """Train a sentiment analysis classifier using NBOW + Logistic regression"""
    
    clf = LogisticRegression(max_iter=200).fit(train_corpus, train_labels)
    
    return clf


def evaluate_sentiment_analysis(classifier, test_corpus, test_labels):
    """Evaluate classifier in the test corpus and report accuracy"""
    
    preds = classifier.predict(test_corpus)
    acc = accuracy_score(test_labels, preds)
    
    return acc


pos_train = read_samples(pos_train_dir, preprocess)
neg_train = read_samples(neg_train_dir, preprocess)

pos_test = read_samples(pos_test_dir, preprocess)
neg_test = read_samples(neg_test_dir, preprocess)

train_corpus, train_labels = create_corpus(pos_train, neg_train)

test_corpus, test_labels = create_corpus(pos_test, neg_test)


# > Results for vectors from Gutenberg Corpus


model = Word2Vec.load("word2vec.model")
nbow_train_corpus = extract_nbow(train_corpus,model,google=False,length=100)

nbow_test_corpus = extract_nbow(test_corpus,model,google=False,length=100)

clf = train_sentiment_analysis(nbow_train_corpus, train_labels)

acc = evaluate_sentiment_analysis(clf, nbow_test_corpus, test_labels)
print('Custom Word2Vec has achieved {:.3f}% accuracy on test set\n'.format(acc*100))


# > Results for GoogleNews vectors


model_google = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True, limit=NUM_W2V_TO_LOAD)


nbow_train_corpus = extract_nbow(train_corpus,model_google,google=True,length=300)

nbow_test_corpus = extract_nbow(test_corpus,model_google,google=True,length=300)

clf = train_sentiment_analysis(nbow_train_corpus, train_labels)

acc = evaluate_sentiment_analysis(clf, nbow_test_corpus, test_labels)
print('GoogleNews Word2Vec has achieved {:.3f}% accuracy on test set'.format(acc*100))
