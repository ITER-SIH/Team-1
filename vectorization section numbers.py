import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

with open("E:/JURIMIND/v0.0.1/ipc_reduced.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

section_no = [str(entry['Section']) for entry in data]
tfidf_vectorizer = TfidfVectorizer()
section_no_vectors = tfidf_vectorizer.fit_transform(section_no)

section_no_output_file = 'E:/vectorization/ipc_section_no_vectors.pkl'
joblib.dump(section_no_vectors, section_no_output_file)


