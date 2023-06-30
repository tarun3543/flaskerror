from flask import Flask, render_template, request, redirect,jsonify
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random

app = Flask(__name__)

# Load login passwords from Excel file
login_check = pd.read_excel(r'C:\Users\tgangadh\Desktop\logincheck.xlsx')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get the entered email address
        email_address = request.form['email_address']

        # Check if the email address is in the DataFrame
        if email_address in login_check['email_address'].tolist():
            # Redirect to the chatbot page
            return redirect('/chatbot')

        # If email address is incorrect, render the login page with an error message
        return render_template('login.html', error=True)

    # Render the login page
    return render_template('login.html', error=False)


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        # Get the user input from the form
        user_input = request.json['user_input']

        # Generate the chatbot response
        response = generate_response(user_input)

        # Return the response as JSON
        return jsonify(response=response)

    # Render the chatbot page without any response
    return render_template('chathtml3.html', response='')

def preprocess_text(text):
    # Implement your own text preprocessing logic here
    return text

def generate_response(prompt):
    df = pd.read_csv(r"C:\chatbot req\Deploy Tech Community GSD.csv")

    # Replace NaN values with empty strings
    df.fillna('', inplace=True)

    df1 = df[['Market Name', 'Sub-Stream', 'SW Release', 'Title', 'Use Case / Problem statement',
              'Use Case/Issue Description ']].copy()
    df1['Combined'] = df1.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    df2 = df[['Work Around']].copy()

    # Reset the indices to match between df1 and df2
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    tfidf_matrix = vectorizer.fit_transform(df1['Combined'])

    # Preprocess the user prompt
    preprocessed_prompt = preprocess_text(prompt)

    keywords = [word.lower() for word in word_tokenize(preprocessed_prompt) if word.lower() not in stop_words]
    keyword_matrix = vectorizer.transform([' '.join(keywords)])
    similarity_scores = cosine_similarity(tfidf_matrix, keyword_matrix)

    match_indices = similarity_scores.argmax(axis=0)  # Find the indices with maximum similarity score
    for index in match_indices:
        if similarity_scores[index] > 0:
            break  # Exit the loop after finding the first match

    if similarity_scores[index] > 0:
        answer_main = df2.loc[index, 'Work Around']
        return answer_main
    elif any(greeting in prompt.lower() for greeting in ['hi', 'hello', 'hey']):
        greetings = ["Hey, how can I help you?", "Hello, how can I assist you?", "Hi there, how may I help you?"]
        return random.choice(greetings)
    elif "who are you" in prompt.lower():
        return "I am a chatbot built to solve RIDB and RPM queries."
    elif "who programmed you" in prompt.lower():
        creators = ["People from RPM team", "The talented minds at RPM", "RPM team members"]
        return "I was created by " + random.choice(creators) + "."
    else:
        alternative_responses = [
            "Apologies, but I don't have an answer to that question.",
            "I'm sorry, that's beyond my knowledge base.",
            "Unfortunately, I don't have the information you're looking for.",
            "Regrettably, I can't provide an answer to your question.",
            "Sorry, I'm unable to assist with that particular inquiry.",
            "I apologize, but I don't have the relevant information."
        ]
        return random.choice(alternative_responses)


if __name__ == '__main__':
    app.run(debug=True)




