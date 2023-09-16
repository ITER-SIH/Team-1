import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

with open("E:/JURIMIND/v0.0.1/ipc_reduced.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

chapter_titles = [(entry['chapter_title']) for entry in data]
tfidf_vectorizer = TfidfVectorizer()
chapter_titles_vectors = tfidf_vectorizer.fit_transform(chapter_titles)

chapter_titles_output_file = 'E:/vectorization/ipc_chapter_titles_vectors.pkl'
joblib.dump(chapter_titles_vectors, chapter_titles_output_file)


