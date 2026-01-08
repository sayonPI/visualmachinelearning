from flask import Flask, request, jsonify
import pickle 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
from pathlib import Path
import pandas as pd
from tensorflow.keras.utils import pad_sequences


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'

#Module-level cached artifacts (load at startup)
model = None 
le_location = None
location_to_idx = None 
max_length = None

def init_artifacts():
    global model, le_location, location_to_idx, max_length

    #Loading model once with compile set to False; saving memory and startup time
    try:
        model = keras.models.load_model(MODEL_DIR / 'location_order_lstm_model.h5')
        with open(MODEL_DIR / 'location_encoder.pkl', 'rb') as f:
            le_location = pickle.load(f)
        with open(MODEL_DIR / 'model_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
            location_to_idx = artifacts['location_to_idx']
            max_length = artifacts['max_length']
        logger.info("Artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")

#Load encoder pickle once 
enc_path = MODEL_DIR / 'location_encoder.pkl'
try: 
    with open(enc_path, 'rb') as f:
        le_location = pickle.load(f)
    logger.info("Location encoder loaded successfully.")
except FileNotFoundError:
    logger.error(f"Location encoder file not found at {enc_path}")
    raise


def load_model():
    global model
    try:
        model = keras.models.load_model("../models/location_order_lstm_model.h5")
        print("Model Loaded Successfully")
    except FileNotFoundError:
        print("Model file not found. Please ensure 'model.pkl' exists.")
    except Exception as e:
        print(f"Error loading model: {e}")
load_model()

@app.route('/')
def index():
    return "It works apparently"

@app.route('/api/predict/itemLoc', methods=['POST'])
def predict_item_locations():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json()
        items = data.get('items')

        if items is None:
            return jsonify({"error": "No items provided"}), 400
        
        itemList = []
        
        for itemlocs in items:
            locations = predict_location_order_lstm(itemlocs)
            itemList.append(locations['LOCATION_ID'].tolist()[0])

        prediction = predict_location_order_lstm(itemList)  

        return jsonify({
            "prediction": prediction['LOCATION_ID'].tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        features = data.get('features')

        if features is None:
            return jsonify({"error": "No features provided"}), 400

        # input_data = np.array(features)
        input_data = features
        print("Input data for prediction:", input_data)

        prediction = predict_location_order_lstm(input_data)

        # loc_list = []
        # for pred in prediction:
        #     print("Predicted location:", pred)
        #     loc_list.append(pred['location'])

        return jsonify({
            "prediction": prediction['LOCATION_ID'].tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def predict_location_order_lstm(location_ids):
    """
    Predict the picking order for a list of location IDs using LSTM
    
    Parameters:
    -----------
    location_ids : list
        List of location IDs for an order
    
    Returns:
    --------
    DataFrame with location IDs and predicted ranks
    """
    # model = keras.models.load_model('location_order_lstm_model.h5', compile=False)

    with open('../models/location_encoder.pkl', 'rb') as f:
        le_location = pickle.load(f)

    print("Location Encoder", le_location.classes_)

    # Encode locations
    encoded_locations = []
    unknown_locations = []
    
    for loc_id in location_ids:
        if loc_id in le_location.classes_:
            encoded_locations.append(le_location.transform([loc_id])[0])
        else:
            # Use 0 for unknown locations (padding value)
            encoded_locations.append(0)
            unknown_locations.append(loc_id)
    
    if unknown_locations:
        print(f"Warning: Unknown locations (will use padding): {unknown_locations}")
    
    # Create sequence for each location position
    predictions = []
    sequence = np.array(encoded_locations)
    max_sequence_length = max(len(seq) for seq in location_ids)

    # Pad sequence
    sequence_padded = pad_sequences([sequence], maxlen=max_sequence_length, padding='post')
    
    # Predict for each position in the sequence
    for i, loc_id in enumerate(location_ids):
        # Create input for this position
        input_seq = np.tile(sequence_padded, (1, 1))
        
        # Predict
        pred_probs = model.predict(input_seq, verbose=0)[0]
        predicted_rank = np.argmax(pred_probs) + 1  # Convert back to 1-indexed
        confidence = pred_probs.max()
        
        predictions.append({
            'LOCATION_ID': loc_id,
            'PREDICTED_RANK': predicted_rank,
            'CONFIDENCE': confidence
        })
    
    # Create result dataframe and sort by predicted rank
    result_df = pd.DataFrame(predictions)
    result_df = result_df.sort_values('PREDICTED_RANK').reset_index(drop=True)
    result_df['SUGGESTED_ORDER'] = range(1, len(result_df) + 1)
    
    return result_df

def predict_next_locations_with_artifacts(model, artifacts, input_sequence, top_k=3):
    """
    Predict the top-k next locations given an input sequence of location IDs.
    """
    location_to_idx = artifacts['location_to_idx']
    idx_to_location = artifacts['idx_to_location']
    max_length = artifacts['max_length']

    # Encode input sequence
    encoded = [location_to_idx.get(loc, 0) for loc in input_sequence]
    # Pad sequence
    padded = pad_sequences([encoded], maxlen=max_length, padding='pre')
    # Predict probabilities
    predictions = model.predict(padded, verbose=0)[0]
    # Get top-k indices
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    # Decode to location IDs
    results = []
    for idx in top_indices:
        if idx in idx_to_location:
            results.append({
                'location': idx_to_location[idx],
                'probability': predictions[idx]
            })
    return results

def load_trained_model():
    """Load the saved model and artifacts"""

     # Load model
    loaded_model = keras.models.load_model('../models/lstm_location_model.keras')
    
    # Load artifacts
    with open('../models/model_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    print("Model and artifacts loaded successfully!")
    print(f"Vocabulary size: {artifacts['vocab_size']}")
    print(f"Max sequence length: {artifacts['max_length']}")
    print(f"Test accuracy: {artifacts['test_accuracy']:.4f}")
    
    return loaded_model, artifacts

# Example usage:
loaded_model, artifacts = load_trained_model()
input_seq = ['D5000000', 'E4000000', 'F8508200', 'C4000000', 'D4000000', 'F4000000', 'D4000000', 'E4000000', 'D4000000', 'F5000000', 'F4000000', 'D8348400', 'G8648100', 'D8308800', 'D6000000', 'G8608200', 'D5000000', 'D5000000']
print("First Model Predictions:")
print(predict_location_order_lstm(input_seq))
# print("Second Model Predictions:")
# print(predict_next_locations_with_artifacts(loaded_model, artifacts, input_seq, top_k=3))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)