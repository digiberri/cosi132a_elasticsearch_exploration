from keybert import KeyBERT


def extract_keywords(doc: str):
    transformer_model = 'nq-distilbert-base-v1'
    keywords_list = KeyBERT(transformer_model).extract_keywords(doc)
    keywords = [word for word, prob in keywords_list]
    return " ".join(keywords)

