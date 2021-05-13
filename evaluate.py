from typing import List

import argparse
from pathlib import Path
from utils import parse_wapo_topics
from metrics import ndcg, Score
from keybert_extraction import extract_keywords
from extract import filter_content


from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Match, ScriptScore, Query
from elasticsearch_dsl.connections import connections
from embedding_service.client import EmbeddingClient

data_dir = Path("data")
xml_path = data_dir.joinpath("topics2018.xml")


def evaluate(index_name: str, query_text: str, query_type: str, k: int = 20, vector_name: str = None,
             using_topic_id: bool = False, using_custom: bool = False):
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")

    if using_topic_id:
        # command line interface accesses queries this way, but the method also
        # supports direct querying via Flask app
        topic_id = query_text
        query_type_index = {'title': 0, 'description': 1, 'narrative': 2, 'narration': 2,
                            'expanded_description': 1, 'keyBERT': 4}[query_type]
        topics = parse_wapo_topics(str(xml_path))
        if query_type == 'keyBERT':
            query_text = " ".join(topics[topic_id])

        else:
            query_text = topics[topic_id][query_type_index]

    if using_custom:
        query = Match(custom_content={"query": query_text})

    elif vector_name:
        if query_type == 'expanded_description':
            query_text = filter_content(query_text)

        elif query_type == 'keyBERT':
            query_text = extract_keywords(query_text)

        encoding_map = {"sbert_vector": "sbert", "ft_vector": "fasttext"}
        encoder = EmbeddingClient(host="localhost", embedding_type=encoding_map[vector_name])
        query_vector = encoder.encode([query_text], pooling="mean").tolist()[0]
        query = generate_script_score_query(query_text, query_vector, vector_name)
    else:
        query = Match(content={"query": query_text})

    return search(index_name, query, k)


def generate_script_score_query(query_text: str, query_vector: List[float], vector_name: str) -> Query:
    """
    generate an ES query that match all documents based on the cosine similarity
    :param query_text: string query
    :param query_vector: query embedding from the encoder
    :param vector_name: embedding type, should match the field name defined in BaseDoc ("ft_vector" or "sbert_vector")
    :return: an query object
    """
    query = ScriptScore(
        query={"match": {"content": query_text}},  # use BM25 with standard analyzer as the base query
        script={  # script your scoring function
            "source": f"cosineSimilarity(params.query_vector, '{vector_name}') + 1.0",
            # add 1.0 to avoid negative score
            "params": {"query_vector": query_vector},
        },
    )
    return query


def search(index: str, query: Query, k: int):
    s = Search(using="default", index=index).query(query)[:k]
    return s.execute()


def print_top_k_hits(hits):
    for hit in hits:
        print(
            hit.meta.id, hit.meta.score, hit.annotation, hit.title, sep="\t"
        )


def list_relevance_judgements(hits, topic_id):
    relevance_judgements = []
    for hit in hits:
        try:
            annotation_split = hit.annotation.split("-")
        except IndexError:
            relevance_judgements.append(0)
        else:
            if annotation_split[0] == topic_id:
                relevance_judgements.append(int(annotation_split[1]))
            else:
                relevance_judgements.append(0)

    return relevance_judgements


def produce_metrics(index_name):
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")
    topics = parse_wapo_topics(str(xml_path))

    for topic in list(topics.items())[:5]:  # first 5 topics (change list indexes to alter)
        topic_id = topic[0]
        # if topic_id == str(690):
        fields = ['title', 'description', 'narrative', 'expanded_description', 'keyBERT']
        bm25_standard_hits = []
        bm25_custom_hits = []
        fasttext_hits = []
        sbert_hits = []

        # populates above lists with lists of Search objects returned by evaluate methods for each field
        for field in fields:
            bm25_standard_hits.append(evaluate(index_name, topic_id, field, using_topic_id=True))
            bm25_custom_hits.append(evaluate(index_name, topic_id, field, using_topic_id=True, using_custom=True))
            fasttext_hits.append(evaluate(index_name, topic_id, field, using_topic_id=True, vector_name='ft_vector'))
            sbert_hits.append(evaluate(index_name, topic_id, field, using_topic_id=True, vector_name='sbert_vector'))

        # maps row title to name of list containing search results for each method
        data_titles_to_lists_map = {'BM25+standard': 'bm25_standard_hits',
                                    'BM25+custom': 'bm25_custom_hits',
                                    'fastText': 'fasttext_hits',
                                    'sBERT   ': 'sbert_hits'}

        # converts each Search object to a list of relevance judgements and passes it to ndcg
        # prints the result for each field and each row title with fancy formatting
        # NOTE: ensure header matches contents of fields list (and in correct order)
        row_format = "{:27}{:8}{:8}{:11}{:8}{:8}{:11}{:8}{:8}{:11}{:8}{:8}{:11}{:8}{:8}{:11}"
        title = "{:27}{:27}{:27}{:27}{:27}{:27}"
        print(title.format('Topic ' + topic_id, 'Title', 'Description', 'Narrative',
                           "Expanded Description", "keyBERT"))
        print("{:27}{:8}{:8}{:11}".format("", "ave_p", "prec", "ndcg"), end="")
        for i in range(len(fields) - 1):
            print("{:8}{:8}{:11}".format("ave_p", "prec", "ndcg"), end="")
        print()
        for data_title, data_list_name in data_titles_to_lists_map.items():
            data_list = eval(data_list_name)
            print_out=[data_title]
            for i in range(len(fields)):
                rel_judge = Score.eval(list_relevance_judgements(data_list[i], topic_id), 20)
                print_out.append(str(round(rel_judge.ap, 5)))
                print_out.append(str(round(rel_judge.prec, 5)))
                print_out.append(str(round(rel_judge.ndcg, 5)))

                # table contents
            print(row_format.format(*print_out))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elasticsearch IR system")
    parser.add_argument("--index_name")
    parser.add_argument("--topic_id")
    parser.add_argument("--query_type")
    parser.add_argument("--top_k", type=int)
    parser.add_argument("-u", action='store_true')
    parser.add_argument("--vector_name")
    parser.add_argument("--produce_metrics", action='store_true')
    args = parser.parse_args()

    if args.produce_metrics:
        produce_metrics(args.index_name)

    else:
        if args.u and args.vector_name:
            print("Use of the custom analyzer with word embeddings is not supported.\n"
                  "Your query will be processed with the standard analyzer and your chosen embeddings.\n")

        if args.index_name and args.topic_id and args.query_type:
            print_top_k_hits(evaluate(index_name=args.index_name, query_text=args.topic_id, query_type=args.query_type,
                             k=args.top_k, vector_name=args.vector_name, using_topic_id=True, using_custom=args.u))
        else:
            print("ERROR: INCOMPLETE FUNCTION CALL\n"
                  "Function call requires the following format (additional arguments optional):\n"
                  "python evaluate.py --index name INDEX_NAME --topic_id TOPIC_ID --query_type QUERY_TYPE")
