from flask import Flask, render_template, request,flash
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib
import csv

app = Flask(__name__)
CSV_FILE = 'music.csv'
@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/learn', methods=['POST','GET'])
def learn():
    if request.method == 'POST':
        # new data from user
        age = request.form['age']
        gender = request.form['gender']
        genre= request.form['genre']
        
        # update the file with new data
        with open(CSV_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([age, gender, genre])
        
        # read the file
        music_dt  =pd.read_csv('music.csv')
        
        # prepare 2 groups and fit the model
        X=music_dt.drop(columns=['genre']) # sample features
        Y=music_dt['genre'] # sample output
        model = DecisionTreeClassifier()
        model.fit(X,Y) # load features and sample data
        # save the model 
        joblib.dump(model, 'our_pridction.joblib') #binary file
        # Flash a success message
        # flash('Data added successfully!', 'success')
        return render_template('learn.html')
    else:
        return render_template('learn.html')



@app.route('/result', methods=['POST'])
def result():
    age = request.form['age']
    gender = request.form['gender']
    model=joblib.load('our_pridction.joblib')
    predictions= model.predict([[age,gender]])
    

    # Process the age and gender data as needed
    return render_template('result.html',  data=predictions)



@app.route('/')
def display_csv():
    # Assuming your CSV file is named music.csv and is in the same directory as this script
    csv_data = pd.read_csv(CSV_FILE)
    headers = csv_data.columns.tolist()
    data = csv_data.values.tolist()

    X=csv_data.drop(columns=['genre']) #age,gender
    Y=csv_data['genre'] # sample output
    model = DecisionTreeClassifier()
    model.fit(X,Y) # load features and sample data
    predictions= model.predict([[22,1],[30,1]]) # make prediction base on the 
    print(predictions)
    return render_template('display.html', headers=headers, data=data)

if __name__ == '__main__':
    app.run(debug=True)
