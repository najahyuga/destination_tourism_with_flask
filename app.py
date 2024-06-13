from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
tourism_data = pd.read_csv('tourism_with_id.csv')
tourism_rating = pd.read_csv('tourism_rating.csv')

# Initialize Flask application
app = Flask(__name__)

# Preprocess the data
tfidf = TfidfVectorizer(stop_words='english')
tourism_data['Description'] = tourism_data['Description'].fillna('')
tfidf_matrix = tfidf.fit_transform(tourism_data['Description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(tourism_data.index, index=tourism_data['Place_Name']).drop_duplicates()

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for recommendation
@app.route('/recommend', methods=['POST'])
def recommend():
    place_name = request.form['place_name']
    try:
        recommendations = get_recommendations(place_name)
        return render_template('recommend.html', place_name=place_name, recommendations=recommendations)
    except KeyError as e:
        return render_template('error.html', error_message=str(e))

def get_recommendations(name):
    try:
        idx = indices[name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        place_indices = [i[0] for i in sim_scores]
        recommended_places = tourism_data.iloc[place_indices]['Place_Name'].tolist()
        return recommended_places
    except KeyError:
        raise KeyError(f"Tempat wisata '{name}' tidak ditemukan dalam dataset.")

if __name__ == '__main__':
    app.run(debug=True)
