from flask import render_template, request, redirect, flash
from app import app
from main import getPrediction

@app.route('/')
def index():
    return render_template('index.html',result="1")


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        l=[]
        l.append(request.form['textarea'])
        label = getPrediction(l)
        flash(label)
        flash(l[0])
        return redirect('/')


if __name__ == "__main__":
    app.run()