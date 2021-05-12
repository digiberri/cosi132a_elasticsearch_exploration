from keybert import KeyBERT
from embedding_service.client import EmbeddingClient
from elasticsearch_dsl.query import Ids,Query
from elasticsearch_dsl import Search

from elasticsearch_dsl.connections import connections
keyword = EmbeddingClient(host="localhost", embedding_type="keybert")
connections.create_connection(hosts=["localhost"], timeout=100, alias="default")

def search(index: str, query: Query) -> None:
    s = Search(using="default", index=index).query(query)[
        :5
    ]  # initialize a query and return top five results
    response = s.execute()
    return response
#first 2 are 690-
q_match_ids = Ids(values=[46517,24370,41488])
res = search(index="wapo_docs_50k",query=q_match_ids)
for x in res:
    print(keyword.encode([x.custom_content]),x.annotation)
