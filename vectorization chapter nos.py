import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

with open("E:/JURIMIND/v0.0.1/ipc_reduced.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

chapter_nos = [str(entry['chapter']) for entry in data]
tfidf_vectorizer = TfidfVectorizer()
chapter_nos_vectors = tfidf_vectorizer.fit_transform(chapter_nos)

chapter_nos_output_file = 'E:/vectorization/ipc_chapter_nos_vectors.pkl'
joblib.dump(chapter_nos_vectors, chapter_nos_output_file)


