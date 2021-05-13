from utils import parse_wapo_topics
from pathlib import Path
from collections import Counter
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
import re
import nltk

data_dir = Path("data")
xml_path = data_dir.joinpath("topics2018.xml")
stemmer = PorterStemmer()
def extract_stops():
    """
    Benjamin Siege
    Extract common stop words from list of topics
    """
    topics = parse_wapo_topics(str(xml_path))
    c = Counter()
    for topic_id in topics:
        topic = topics[topic_id]
        for x in topic:
            c.update(nltk.word_tokenize(x))
    out = []
    for item in c.most_common(30):
        # if item[1] > 10:
        out.append(stemmer.stem(item[0]))
    return out

#Borrowed from Stack Overflow discussion of synsets with POStags
#Michael's edits
def pos_to_wordnet_pos(pt, returnNone=False):
   ' Mapping from POS tag word wordnet pos tag '
   tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,'VB':wn.VERB, 'RB':wn.ADV}
   try:
       return tag[pt[:2]]
   except:
       return None if returnNone else ''
    

def filter_content(query,n=-1, m=-1):
    """
    Benjamin Siege and Michael Gardner
    Use NLTK POS tagging to extract contentful words
    """
    def content(pos_tup):
        stop_words = list(extract_stops())
        # Match nouns, adverbs, adjectives and verbs
        regex = "(NN|JJ).{0,2}$"
        #return re.match(regex,pos_tup[1])
        
        if re.match(regex,pos_tup[1]):
            return not stemmer.stem(pos_tup[0]) in stop_words
        return False
        
    tokens = nltk.word_tokenize(query)
    out_tokens = list(filter(content,nltk.pos_tag(tokens)))
    out = []
    for token,tag in out_tokens:
        # print(token)
        synsets = wn.synsets(token, pos_to_wordnet_pos(tag))
        if synsets:
            lemN = synsets[n%len(synsets)].lemma_names()
            out.append(lemN[m%len(lemN)])
        out.append(token)
    return " ".join(out).replace("_"," ")




if __name__ == "__main__":
    print(filter_content("Find documents which describe an advantage in hiring potential or increased income for graduates of U.S. colleges."))
    print(filter_content("Relevant documents cite some advantage of a college education for job opportunities. Documents citing better opportunities for non-college vocational-training is not relevant."))
