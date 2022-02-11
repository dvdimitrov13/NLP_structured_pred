from flask import request
from flask import jsonify
from flask import Flask, render_template

from predict import predict, extract_ingredient, process_string, f1_multiclass

app = Flask(__name__)


@app.route("/")
def my_form():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def my_form_post():
    text = request.form["text"]
    output = "I think this recipe might be {}!".format(predict(text))
    return render_template("index.html", variable=output)


if __name__ == "__main__":
    app.run(port="8088", threaded=False)