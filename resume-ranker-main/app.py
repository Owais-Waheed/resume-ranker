from flask import Flask, render_template, request, send_file
import spacy
import PyPDF2
from sentence_transformers import SentenceTransformer, util
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")
wn.ensure_loaded()


# Load the Sentence Transformer model
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer('all-mpnet-base-v2')
# model = SentenceTransformer('all-roberta-large-v1')

# Initialize results variable
results = []

def remove_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)

# # Extract text from PDFs
# def extract_text_from_pdf(pdf_path):
#     with open(pdf_path, "rb") as pdf_file:
#         pdf_reader = PyPDF2.PdfReader(pdf_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#         text = remove_stopwords(text)
#         text = lemmatize_text(text)
#         return text

def extract_text_from_pdf(pdf_path, job_keywords):
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    return False  # Return False if any page text extraction fails

            if '\n' not in text:
                return False  # Return False if text does not contain any newline characters
            
            
            matched_words = [word for word in job_keywords.split(",") if word.lower() in text.lower()]

            text = remove_stopwords(text)
            text = lemmatize_text(text)
            return text, matched_words
    except (PyPDF2.errors.PdfReadError, FileNotFoundError, IOError, Exception) as e:
        # Handle specific and general exceptions that might occur while reading the PDF
        print(f"Error reading PDF file: {e}")
        return False

# Extract entities using spaCy NER
def extract_entities(text):
    # emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    email_pattern = r'[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\s*\.\s*[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)

    # Clean up the emails by removing spaces
    emails = [email.replace(' ', '') for email in emails]
    names = re.findall(r'\b[A-Z][a-z]*\s[A-Z][a-z]*\b', text)
    return emails, names

@app.route('/', methods=['GET', 'POST'])
def index():
    global results
    results = []
    if request.method == 'POST':
        job_description = request.form['job_description']
        job_keywords = request.form['job_keywords']
        job_description = remove_stopwords(job_description)
        job_description = lemmatize_text(job_description)

        # print(f"Job Description: {job_description}")

        resume_files = request.files.getlist('resume_files')

        # Create a directory for uploads if it doesn't exist
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        # Process uploaded resumes
        processed_resumes = []
        for resume_file in resume_files:
            resume_path = os.path.join("uploads", resume_file.filename)
            resume_file.save(resume_path)
            matched_words = []
            resume_text , matched_words = extract_text_from_pdf(resume_path, job_keywords.lower())
            if not resume_text:
                continue

            # print(f"Resume Text: {resume_text}")

            emails, names = extract_entities(resume_text)
            processed_resumes.append((names, emails, resume_text, matched_words))

        # Calculate similarity using Sentence Transformer
        job_desc_vector = model.encode(job_description + " " + job_keywords)
        ranked_resumes = []
        for (names, emails, resume_text, matched_words) in processed_resumes:
            resume_vector = model.encode(resume_text)
            resume_text = resume_text.lower()
            match_words = ""
            for words in matched_words:
                match_words =  words + "," + match_words
            similarity = util.pytorch_cos_sim( job_desc_vector, resume_vector).item() * 100
            similarity = round(similarity, 2)
            ranked_resumes.append((names, emails, similarity, match_words))

        # Sort resumes by similarity score
        ranked_resumes.sort(key=lambda x: x[2], reverse=True)
        results = ranked_resumes

    return render_template('index.html', results=results)

@app.route('/download_csv')
def download_csv():
    csv_content = "Rank,Name,Email,Similarity,Matched_keywords\n"
    for rank, (names, emails, similarity,matched_words) in enumerate(results, start=1):
        name = names[0] if names else "N/A"
        email = emails[0] if emails else "N/A"
        match_words = matched_words if matched_words else "N/A"
        # print(f"Name: {name}, Email: {email}, Similarity: {similarity}, Matched Words: {match_words}")
        csv_content += f"{rank},{name},{email},{similarity},{match_words}\n"

    csv_filename = "ranked_resumes.csv"
    with open(csv_filename, "w") as csv_file:
        csv_file.write(csv_content)

    csv_full_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), csv_filename)
    return send_file(csv_full_path, as_attachment=True, download_name="ranked_resumes.csv")

if __name__ == '__main__':
    app.run(debug=True)
