from evaluate import *
from pathlib import Path
import argparse,math
import shelve,time
from flask import Flask, render_template, request, Markup
from elasticsearch_dsl.query import Match, ScriptScore, Query, MatchAll,Ids
from elasticsearch_dsl.connections import connections
app = Flask(__name__)
# Flask code From Benjamin Siege

# home page
@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["POST"])
def results():

    # TODO: Use radio button values to select type
    text = request.form["query"]
    q_type = request.form["qtype"]
    if q_type == "BM25":
        results = evaluate(index_name="wapo_docs_50k",query_text=text,query_type="",k=100)
    elif q_type =="BM25_custom":
        results = evaluate(index_name="wapo_docs_50k",query_text=text,query_type="",using_custom=True,k=100)
    elif q_type == "expanded_description":
        results = evaluate(index_name="wapo_docs_50k",query_text=text,query_type=q_type,k=100,vector_name="sbert_vector")
    elif q_type == "keyBERT":
        results = evaluate(index_name="wapo_docs_50k",query_text=text,query_type=q_type,k=100,vector_name="sbert_vector")
    else:
        results = evaluate(index_name="wapo_docs_50k",query_text=text,query_type=q_type,k=100,vector_name=q_type)
    matches = []
    for hit in results:
        matches.append((hit.title,hit.meta.id,round(hit.meta.score,4),hit.content[:150]))
    return render_template("results.html",matches=matches[:min(8,len(matches))],query=text,maxpages=math.ceil(len(matches)/8),qtype=q_type)


# "next page" to show more results
@app.route("/results/<int:page_id>", methods=["POST"])
def next_page(page_id):
    text = request.form["query"]
    q_type = request.form["qtype"]
    if q_type == "BM25":
        results = evaluate(index_name="wapo_docs_50k",query_text=text,query_type="",k=100)
    elif q_type =="BM25_custom":
        results = evaluate(index_name="wapo_docs_50k",query_text=text,query_type="",using_custom=True,k=100)
    elif q_type == "expanded_description":
        results = evaluate(index_name="wapo_docs_50k",query_text=text,query_type=q_type,k=100,vector_name="sbert_vector")
    elif q_type == "keyBERT":
        results = evaluate(index_name="wapo_docs_50k",query_text=text,query_type=q_type,k=100,vector_name="sbert_vector")
    else:
        results = evaluate(index_name="wapo_docs_50k",query_text=text,query_type=q_type,k=100,vector_name=q_type)
    for hit in results:
        matches.append((hit.title,hit.meta.id,round(hit.meta.score,4),hit.content[:150]))
    return render_template("results.html",matches=matches[8*page_id:min(8*(page_id+1),len(matches))],query=text,maxpages=math.ceil(len(matches)/8),qtype=q_type)


# document page
@app.route("/doc_data/<int:doc_id>")
def doc_data(doc_id):
    # TODO:
    result = search("wapo_docs_50k",Ids(values=[doc_id]),40)
    return render_template("doc.html",doc=result[0].to_dict(),text=Markup(result[0].content))

if __name__ == "__main__":
    try:
        connections.create_connection(hosts=["localhost"],alias="default")
    except Exception as e:
        print(f"encountered exception {e}")
        print("please make sure the elasticsearch server and any required embeddings servers are running")
    app.run(debug=True, port=5000)
