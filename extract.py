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


def filter_content(query):
    """
    Benjamin Siege
    Use NLTK POS tagging to extract contentful words
    """
    def content(word):
        pos_tup = nltk.pos_tag([word])[0]
        stop_words = list(extract_stops())
        # Match nouns, adverbs, adjectives and verbs
        regex = "(NN|JJ).{0,2}$"
        if re.match(regex,pos_tup[1]):
            return not stemmer.stem(pos_tup[0]) in stop_words
        return False
    tokens = nltk.word_tokenize(query)
    out_tokens = list(filter(content,tokens))
    out = []
    for token in out_tokens:
        # print(token)
        synsets = wn.synsets(token)
        if synsets:
            out.extend(synsets[0].lemma_names())
        out.append(token)
    return " ".join(out).replace("_"," ")


if __name__ == "__main__":
    print(filter_content("Find documents which describe an advantage in hiring potential or increased income for graduates of U.S. colleges."))
    print(filter_content("Relevant documents cite some advantage of a college education for job opportunities. Documents citing better opportunities for non-college vocational-training is not relevant."))
