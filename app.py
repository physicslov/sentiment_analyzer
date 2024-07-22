from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('sentiment_model.h5')

# Load the tokenizer
with open('token_1.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_review(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    return padded_sequence

def predict_sentiment(review):
    preprocessed_review = preprocess_review(review)
    sentiment = model.predict(preprocessed_review)[0][0]
    if sentiment > 0.5:
        return 'Positive'
    else:
        return 'Negative'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        prediction = predict_sentiment(review)
        return render_template('index.html', prediction_text=f'Sentiment: {prediction}')
    return None

if __name__ == '__main__':
    app.run(debug=True)
