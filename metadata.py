import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


with open("E:/JURIMIND/v0.0.1/ipc_reduced.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

section_desc = [entry['section_desc'] for entry in data]
section_title = [entry['section_title'] for entry in data]
chapter_title = [entry['chapter_title'] for entry in data]
section_no = [str(entry['Section']) for entry in data] 
chapter_no = [str(entry['chapter']) for entry in data] 

tfidf_vectorizer = TfidfVectorizer()
section_desc_vectors = tfidf_vectorizer.fit_transform(section_desc)
section_title_vectors = tfidf_vectorizer.fit_transform(section_title)
chapter_title_vectors = tfidf_vectorizer.fit_transform(chapter_title)



metadata_df = pd.DataFrame({
    'Section': section_no,
    'chapter': chapter_no,
    'Section_Desc_Vector': section_desc_vectors,
    'Section_Title_Vector': section_title_vectors,
    'Chapter_Title_Vector': chapter_title_vectors
})


metadata_output_file = 'E:/vectorization/ipc_metadata.pkl'
joblib.dump(metadata_df, metadata_output_file)
