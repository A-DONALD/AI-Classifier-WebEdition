from flask import Flask, render_template
import webbrowser

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')


def main():
    url = f"http://localhost:5000"
    webbrowser.open(url)
    app.run(debug=False, port='5000')
