from flask import Flask, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    news = db.relationship('News', backref='user', lazy=True)

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.String(255), nullable=False)

    def __init__(self, user_id, content):
        self.user_id = user_id
        self.content = content

with app.app_context():
    db.create_all()

model = pickle.load(open('model.pkl', 'rb'))
cv = CountVectorizer()

# Load and preprocess the training data
df = pd.read_csv('deceptive-opinion.csv')
df1 = df[['deceptive', 'text']]
df1.loc[df1['deceptive'] == 'deceptive', 'deceptive'] = 0
df1.loc[df1['deceptive'] == 'truthful', 'deceptive'] = 1
X = df1['text']
Y = np.asarray(df1['deceptive'], dtype=int)

# Create and fit CountVectorizer to the training data
cv = CountVectorizer()
X_train = cv.fit_transform(X)

# Train the model using X_train and Y
# (Add your model training code here)

# Save the trained model and CountVectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(cv, open('count_vectorizer.pkl', 'wb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/input')
        else:
            return render_template('login.html', error='Invalid user')

    return render_template('login.html')

@app.route('/input', methods=['GET', 'POST'])
def input():
    if session.get('email'):
        user = User.query.filter_by(email=session['email']).first()

        if request.method == 'POST':
            entered_info = request.form.get('enteredinfo')

            # Check if the entered information is not empty
            if entered_info:
                # Check if entered_info exists in the news table
                news_entry = News.query.filter_by(content=entered_info).first()
                if news_entry:
                    result = 'True'
                else:
                    result = 'Fake'

                return render_template('result.html', result=result)

        return render_template('input.html', user=user)

    return redirect('/login')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/SignUp')
def SignUp():
    return render_template('register.html')

@app.route('/Home')
def Home():
    return render_template('input.html')

@app.route('/AddNews', methods=['GET', 'POST'])
def AddNews():
    if request.method == 'POST':
        if session.get('email'):
            user = User.query.filter_by(email=session['email']).first()

            # Get the entered news content from the form
            news_content = request.form.get('enteredinfo')

            # Check if the news content is not empty
            if news_content:
                # Create a new News instance and add it to the user's news
                new_news = News(user_id=user.id, content=news_content)
                db.session.add(new_news)
                db.session.commit()  # Commit changes to the database

                # Redirect to a page or do something else after storing the news
                return redirect('/Home')

    return render_template('News.html')



@app.route('/adminlogin',methods=['GET','POST'])
def adminlogin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        
        
        if email == "System123@gmail.com" and password == "System@123":
            
            
            return redirect('/AddNews')
        else:
            return render_template('adminlogin.html',error='Invalid user')

    return render_template('adminlogin.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    df = pd.read_csv('deceptive-opinion.csv')
    df1 = df[['deceptive', 'text']]
    df1.loc[df1['deceptive'] == 'deceptive', 'deceptive'] = 0
    df1.loc[df1['deceptive'] == 'truthful', 'deceptive'] = 1
    X = df1['text']
    Y = np.asarray(df1['deceptive'], dtype = int)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=109)
    cv = CountVectorizer()
    x = cv.fit_transform(X_train)
    y = cv.transform(X_test)
    message = request.form.get('enteredinfo')
    if not message:
        return render_template('input.html',  error_message='Please enter information.')
    data = [message]
    vect = cv.transform(data).toarray()
    pred = model.predict(vect)

    return render_template('result.html', prediction_text=pred)

if __name__ == '__main__':
    app.run(debug=True)