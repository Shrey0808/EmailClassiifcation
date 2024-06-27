from flask import Flask , render_template , url_for , request
import joblib
model = joblib.load('BNB_Model.lb')
vectorizer = joblib.load('countvectorizer.lb')

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict' , methods = ['GET' , 'POST'])
def predict():
    if request.method == 'POST':
        email_message = str(request.form['email_message'])
        email = [email_message]
        transformed_email = vectorizer.transform(email)
        prediction = str(model.predict(transformed_email)[0])
        dt = {'0':'ham','1':'spam'}
        label = dt[prediction]
        with open('email.txt','w+') as f:
            f.write(f'{label}\t {email_message}\n')
        return label
    
if __name__ == '__main__':
    app.run(debug = True)
