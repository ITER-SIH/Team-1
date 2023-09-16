import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy

with open("E:/JURIMIND/v0.0.1/ipc_reduced.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

    section_descriptions =[entry['section_desc'] for entry in data]
    tfidf_vectorizer = TfidfVectorizer();
section_desc_vectors = tfidf_vectorizer.fit_transform(section_descriptions)

section_desc_output_file = 'E:/vectorization/ipc_section_desc_vectors.pkl'
joblib.dump(section_desc_vectors, section_desc_output_file)


