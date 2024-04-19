from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import sqlite3
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re
import os
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid=SentimentIntensityAnalyzer()



app = Flask(__name__)
app.secret_key = 'sunil_123'

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

global movie_inp

# Database initialization
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            age INT NOT NULL,
            gender TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()



@app.route('/input', methods=['POST'])
def user_input():
    global movie_inp
    session['movie_input'] = request.form['movie_input']
    #print(session['movie_input'])
    movie_inp = session['movie_input']
    return movie_inp




@app.route('/')
def index():
    return render_template('index.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        if user and user[2] == password:
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
    
    return render_template('login.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        age = request.form['age']
        gender = request.form['gender']
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password, age, gender) VALUES (?, ?, ?, ?)", (username, password, age, gender))
            conn.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose a different username.', 'error')
        finally:
            conn.close()
    return render_template('register.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/logout')
def logout():
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))


@app.route('/',methods=['GET'])
def back():
    return redirect(url_for('index'))



@app.route('/collaborative',methods=['GET'])
def collaborative_filtering():
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    df=pd.merge(movies,ratings,on='movie_id')
    df_unique_movies = df.drop_duplicates(subset='name', keep='first')
    scores=pd.merge(movies,ratings,on='movie_id')
    scores.head()
    user_movie_matrix = ratings.pivot(index='user_id', columns='movie_id', values='ratings')
    user_movie_matrix = user_movie_matrix.fillna(0)
    ratings_data = user_movie_matrix.to_numpy()
    X_train, X_test = train_test_split(ratings_data,train_size=0.8, test_size=0.2, random_state=42)
    num_features = ratings_data.shape[1]
    input_layer = Input(shape=(num_features,))
    encoded = Dense(128, activation='relu')(input_layer)
    decoded = Dense(num_features, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(X_train, X_train,epochs=10,batch_size=256,shuffle=True,validation_data=(X_test, X_test))

    # Predict ratings using the trained autoencoder
    ratings_prediction = autoencoder.predict(ratings_data)

    # Create a DataFrame for the predicted ratings
    denoised_ratings_matrix = pd.DataFrame(ratings_prediction, index=user_movie_matrix.index, columns=user_movie_matrix.columns)

    # Scale up the predicted ratings for better differentiation
    predicted_ratings = denoised_ratings_matrix*10

    # Calculate user similarity
    user_similarity = cosine_similarity(predicted_ratings)
    user_similarity_df = pd.DataFrame(user_similarity, index=predicted_ratings.index, columns=predicted_ratings.index)

# Function to get movie ID based on title
    def get_movie_id(movie_title, movies_df):
        regex = re.compile(movie_title, re.I)
        matched_titles = movies_df['name'].str.contains(regex)
        movie_id = movies_df[matched_titles]['movie_id'].iloc[0]
        return movie_id

    # Function to recommend movies based on a given title
    def recommend_movies(movie_title, movies_df, predicted_ratings, user_similarity_df, top_n=10):
        movie_id = get_movie_id(movie_title, movies_df)
        movie_ratings = predicted_ratings[movie_id]
        similarity_scores = user_similarity_df.dot(movie_ratings) / movie_ratings.sum()
        sorted_scores = similarity_scores.sort_values(ascending=False)
        top_users = sorted_scores.head(top_n).index
        recommended_movies = user_movie_matrix.loc[top_users].mean().sort_values(ascending=False).head(top_n)
        recommended_movie_ids = recommended_movies.index
        recommended_movie_titles = movies_df[movies_df['movie_id'].isin(recommended_movie_ids)]

        # Include predicted ratings in the output
        recommended_movie_titles['predicted_rating'] = 10*(recommended_movies.loc[recommended_movie_ids].values)
        return recommended_movie_titles[['name', 'predicted_rating']]

    # Get user input for movie title

    # Get recommendations
    recommendations = recommend_movies(movie_inp, movies, predicted_ratings, user_similarity_df)
    recomm=recommendations.values.tolist()

# Display the recommended movies and their predicted ratings
    return recomm
   
@app.route('/content-based',methods=['GET'])
def content_based():
    

    dataset=pd.read_csv('movies.csv',header=None)

    dataset.head()

    dataset[2][1].split('|')

    genres=[]
    for i in dataset[2]:
     temp=i.split('|')
     temp=' '.join(temp)
     genres.append(temp)

    genres

    dataset[1][1].split(' (')

    movies=[]

    for i in dataset[1]:
     temp=i.split(' (')
     temp=''.join(temp[0])
     movies.append(temp)

    movies

    dataset['Movies' ] = pd.DataFrame(movies)

    dataset['Genres'] = pd.DataFrame(genres)

    dataset.head()

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(dataset['Genres'])

    from sklearn.metrics.pairwise import cosine_similarity
    cosine_similarities = cosine_similarity(X , X)

    X.shape

    cosine_similarities.shape

    movie_title = dataset['Movies']

    movie_title
    

    indices = pd.Series(dataset.index , movie_title)

    indices

    list(enumerate(cosine_similarities[1]))

# from sklearn.metrics.pairwise import cosine_similarity
    
    def movie_recommender(matches):
        index = indices[matches]
        similarity_score = list(enumerate(cosine_similarities[index]))
        similarity_score = sorted(similarity_score , key=lambda x:x[1] , reverse=True)
        similarity_score = similarity_score[1:15]
        movie_indices = [i[0] for i in similarity_score]
        return movie_title.iloc[movie_indices]

    movie_recommender('Hush')
    
    recom=movie_recommender(movie_inp).values.tolist()
    return recom

@app.route('/sentiment',methods=['GET'])
def sentiment_analysis():


    movies = pd.read_csv('movies.csv')
    movies.columns = ['movie_id', 'movie_title', 'genres']
    tweet = pd.read_csv('tweets.csv')
    tweet.columns = ['movie_id', 'tweets']

    df = pd.merge(movies, tweet, on='movie_id')

    def sentiment_score(tweets):
        return sid.polarity_scores(tweets)['compound']
    df['sentiment_score'] = df['tweets'].apply(sentiment_score)
    recommended_movies = df[df['sentiment_score'] > 0.5]['movie_title'].unique()
    #print("Recommended Movies based on Positive Sentiments:")
    print(recommended_movies)
    sentiment_list = recommended_movies.tolist()
    return sentiment_list

@app.route('/hybrid', methods=['GET'])
def Hybrid_rec():
    global movie_inp
    global sorted_hybrid_rec

    # Collaborative Filtering Recommendations
    collaborative_rec = collaborative_filtering()

    # Content-Based Filtering Recommendations
    content_based_rec = content_based()

    # Sentiment-Based Recommendations
    sentiment_rec = sentiment_analysis()

    # Assign weights to each recommendation method
    collaborative_weight = 0.25
    content_based_weight = 0.25
    sentiment_weight = 0.50

    # Combine Recommendations using weighted fusion
    hybrid_rec = {}

    # Add collaborative recommendations with weight
    for movie, score in collaborative_rec:
        hybrid_rec[movie] = hybrid_rec.get(movie, 0) + collaborative_weight * score

    # Add content-based recommendations with weight
    for movie in content_based_rec:
        hybrid_rec[movie] = hybrid_rec.get(movie, 0) + content_based_weight

    # Add sentiment-based recommendations with weight
    for movie in sentiment_rec:
        hybrid_rec[movie] = hybrid_rec.get(movie, 0) + sentiment_weight

    # Sort the hybrid recommendations by score
    sorted_hybrid_rec = sorted(hybrid_rec.items(), key=lambda x: x[1], reverse=True)

    # Return the top recommended movies
    return jsonify(sorted_hybrid_rec)



@app.route('/graph',methods=['GET'])
def graphs():
    global sorted_hybrid_rec
    
    # Get the top 10 movies recommended by hybrid filtering
    top_hybrid_rec = sorted_hybrid_rec[:10]
    
    # Extracting movie titles and scores for hybrid recommendations
    hybrid_movies = [movie[0] for movie in top_hybrid_rec]
    hybrid_scores = [score[1] for score in top_hybrid_rec]

    # Plotting the data
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot for hybrid recommendations
    ax.barh(hybrid_movies, hybrid_scores, color='green')
    ax.set_xlabel('Scores')
    ax.set_ylabel('Movies')
    ax.set_title('Top 10 Hybrid Recommendations')
    ax.invert_yaxis()  # Invert y-axis to show highest score at the top

    # Adjust layout
    plt.tight_layout()

    # Saving the plot to the static directory
    plot_path = os.path.join(app.static_folder, 'recommendations_comparison.png')
    plt.savefig(plot_path)

    # Return the plot path
    return render_template('graph.html', plot_path='/static/recommendations_comparison.png')
if __name__ == '__main__':
    app.run(debug=True)


