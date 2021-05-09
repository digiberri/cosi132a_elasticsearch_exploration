from flask import Flask, render_template

app = Flask(__name__)


# home page
@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["POST"])
def results():
    # TODO:
    raise NotImplementedError


# "next page" to show more results
@app.route("/results/<int:page_id>", methods=["POST"])
def next_page(page_id):
    # TODO:
    raise NotImplementedError


# document page
@app.route("/doc_data/<int:doc_id>")
def doc_data(doc_id):
    # TODO:
    raise NotImplementedError


if __name__ == "__main__":
    app.run(debug=True, port=5000)
