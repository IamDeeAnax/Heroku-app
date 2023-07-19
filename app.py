from flask import Flask, request, render_template, jsonify, redirect, url_for
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import nltk
import re
import string
import requests
import zipfile
from nltk.corpus import stopwords
import wordninja
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define a function to preprocess the data
def preprocess_text(text):
    # Remove newlines
    text = re.sub(r'\n+', ' ', text)
    # Remove hyphens and put spaces
    text = re.sub(r'-', ' ', text)
    # Remove words containing numbers
    text = re.sub(r'\b\w*\d\w*\b', ' ', text)
    # Replace one or two letter words with an empty string
    text = re.sub(r'\b\w{1,2}\b', '', text)
    # Remove Roman numerals
    text = re.sub(r'\b[IVXLCDM]+\b', ' ', text, flags=re.IGNORECASE)
    # Convert to lowercase
    text = text.lower()
    # Separate joined words
    text = ' '.join(wordninja.split(text))
    # Remove URLs
    text = re.sub(r'http\S+', ' ', text)
    # Remove any special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    # Replace duplicate word with single word
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
    # Remove punctuation
    text = re.sub(r'[^\w\s]|_', ' ', text)
    # Remove specific words
    text = re.sub(r'\b(?:one|two|use|also|would|first|fig|may|used|see|new|differennt|called|many|find|part|number|using|work|chapter|example|must|true|cos|false|within|result|much|another|figure|form|three|like|however|given)\b', " ", text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:oh|ost|coo|coa|syn|yl|lih|gre|sni|tait|al|ce|ten|elo|oid|ley|rer|se|isra|blu|lk|lu|ree|lt|lus|lu|el|line|thus|end|process|change|different|could)\b', '', text, flags=re.IGNORECASE)
    # Remove single alphabets excluding "a"
    text = re.sub(r"(?<![a-zA-Z])[^aA\s][^a-zA-Z]?(?![a-zA-Z])", "", text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    if request.referrer and request.referrer.endswith(url_for('predict')):
        # User has refreshed the page or navigated back from prediction
        return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.form['text']

    # Check if the text box is empty
    if data.strip() == "":
        return render_template('index.html', prediction_text='Please enter a text.')

    num_classes = 4
    embedding_dim = 50
    embedding_layer_1 = hub.KerasLayer(
        "https://tfhub.dev/google/nnlm-en-dim50/2",
        input_shape=[],
        dtype=tf.string,
        trainable=True,
    )

    # Local paths where the files are saved
    model_path = './models-nnlm50/subject_classification_model_weights.h5'
    encoder_path = './models-nnlm50/encoder_classes.npy'

        # Instantiate your model architecture
    model_load = tf.keras.Sequential([
        embedding_layer_1,
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=num_classes, activation='softmax')
    ])

    # Load the weights from the local path
    try:
        model_load.load_weights(model_path)
        encoder_classes = np.load(encoder_path, allow_pickle=True)
        encoder = LabelEncoder()
        encoder.classes_ = encoder_classes
    except Exception as e:
        error_message = f"Error loading the model: {str(e)}"
        print(error_message)
        return render_template('index.html', prediction_text=error_message)

    # Preprocess the text
    preprocessed_text = preprocess_text(data)

    # Make sure the preprocessed text is not empty
    if preprocessed_text.strip() == "":
        return render_template('index.html', prediction_text='Please enter a valid text.')

    # Remove stop words from the preprocessed text
    filtered_sentence = " ".join([word for word in preprocessed_text.split() if word.lower() not in stop_words])

    # Convert preprocessed text to tensor
    input_sequences = tf.constant([filtered_sentence], dtype=tf.string)

    # Make prediction using model loaded from disk as per the data.
    prediction = model_load.predict(input_sequences)

    # Get the confidence of the prediction
    confidence = np.max(prediction)

    # Set a confidence threshold
    confidence_threshold = 0.5

    # If the model's confidence in its prediction is less than the threshold
    if confidence < confidence_threshold:
        return render_template('index.html', prediction_text='The input text does not belong to any category.')

    # Get the predicted label
    predicted_label = np.argmax(prediction, axis=-1)
    predicted_class_name = encoder.inverse_transform(predicted_label)[0]

    # Take the first value of prediction
    output = 'The predicted school subject is: {}'.format(predicted_class_name)

    # Clear the text area and render the template with the prediction result
    return render_template('index.html', prediction_text=output, error_message=None, input_text=data)

if __name__ == '__main__':
    app.run(port=5000, debug=True)


