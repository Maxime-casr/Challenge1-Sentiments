from flask import Flask, render_template, request
from analysis_engine import analyse_complete

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/analyse", methods=["POST"])
def analyse():
    avis = request.form.get("avis")
    results = analyse_complete(avis)

    return render_template("resultats.html", **results, avis=avis)

if __name__ == "__main__":
    app.run(debug=True)

