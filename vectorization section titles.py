import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

with open("E:/JURIMIND/v0.0.1/ipc_reduced.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

section_titles = [(entry['section_title']) for entry in data]
tfidf_vectorizer = TfidfVectorizer()
section_titles_vectors = tfidf_vectorizer.fit_transform(section_titles)

section_titles_output_file = 'E:/vectorization/ipc_section_titles_vectors.pkl'
joblib.dump(section_titles_vectors, section_titles_output_file)


