from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle as pkl

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/car-price-prediction")
def carpriceprediction():

    dataset = pd.read_csv("cleaned_data.csv")
    companies = sorted(dataset["company"].unique())
    names = sorted(dataset["name"].unique())

    return render_template("carpriceprediction.html", companies = companies, names = names)

@app.route("/car-price-prediction-result")
def carpricepredictionresult():
    name = request.args.get("name")
    company = request.args.get("company")
    year = request.args.get("year")
    kms_running = request.args.get("kms_running")
    fuel_type = request.args.get("fuel_type")

    columns = ["name", "company", "year", "kms_driven", "fuel_type"]
    data = [name, company, year, kms_running, fuel_type]
    myinput = pd.DataFrame(columns = columns, data = np.array(data).reshape(1,5))

    pipe = pkl.load(open('LinearRegressionModel.pkl', 'rb'))

    result = int(pipe.predict(myinput)[0][0])

    return render_template("carpricepredictionresult.html", company = company, name = name, year = year, kms_running = kms_running, fuel_type = fuel_type, result = result)

@app.route("/contact")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug = True, port = 5000)