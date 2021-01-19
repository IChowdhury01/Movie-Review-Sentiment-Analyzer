# Binary Sentiment Analysis using Recurrent Neural Networks

# Import libraries & dataset list
import tensorflow as tf                 
import tensorflow_datasets as dslist

# Load Dataset

print("\nLoading dataset...")
# Download dataset and dataset info
DATASET_CODE = 'imdb_reviews/subwords8k'        # Using a TensorFlow binary sentiment classification dataset
dataset, dsinfo = dslist.load(DATASET_CODE,    
                              with_info=True, 
                              as_supervised=True)    


# Separate into training and testing data.
training = dataset['train']
testing = dataset['test']

# Declare encoder (maps each word in a string to its index in the dataset's vocabulary)
encoder = dsinfo.features['text'].encoder

print("Dataset loaded.")

# Setup for training
# Prepare data. Create batches of encoded strings and zero-pad them.
BUFFER_SIZE = 10000
BATCH_SIZE = 64     # Max number of encoded strings in batch
padded_shapes = ([None], ())

training = (training
            .shuffle(BUFFER_SIZE)
            .padded_batch(BATCH_SIZE, padded_shapes=padded_shapes))


testing = (testing
          .padded_batch(BATCH_SIZE, padded_shapes=padded_shapes))



# Setup Recurrent Neural Network (RNN)
# Create RNN model using Keras.
OUTPUT_SIZE = 64
rnn_model = tf.keras.Sequential([   # Keras Sequential model: processes sequence of encoded strings (indices), embeds each index into vector, then processes through embedding layer
    tf.keras.layers.Embedding(encoder.vocab_size, OUTPUT_SIZE),  # Add embedding layer: stores each word as trainable vector
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),    # Make input sequence iterate both directions through LTSM layer (helps learn long-range dependencies).
    
    # Add layers
    tf.keras.layers.Dense(64, activation='relu'),   
    tf.keras.layers.Dense(1)
])

# Compile RNN model
rnn_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])




# Train RNN
NUM_ITERATIONS = 1

print("\nTraining neural network...")

history = rnn_model.fit(training, epochs=NUM_ITERATIONS, validation_data=testing)

print("Training complete.")



# Test RNN. 
print("\nTesting on dataset...")

loss, accuracy = rnn_model.evaluate(testing)    # Return test loss and test accuracy

print("Testing complete.")



# Process and print results
loss = round(loss, 3)
accuracy = round(accuracy*100, 2)

print("Test Loss: {}".format(loss))
print("Test Accuracy: {}%".format(accuracy))



# Prediction
# Zero-pads a vector up to a target size.
def pad_vector(vec, target_size):
    num_zeros = [0] * (target_size - len(vec))
    vec.extend(num_zeros)
    return vec

# Predicts sentiment. Output will be a decimal number.
def predict_sentiment(review):
    encoded_review = encoder.encode(review) # Encode review. Map each word to an index.

    encoded_review = pad_vector(encoded_review, BATCH_SIZE) # Zero-padding

    encoded_review = tf.cast(encoded_review, tf.float32)
    
    prediction = rnn_model.predict(tf.expand_dims(encoded_review, 0))

    return prediction

# Predictions with value over 0.5 are positive sentiments.
def interpret_prediction(prediction):
    if prediction >= 0.5:
        return "Positive"
    else:
        return "Negative"


# Predict sentiment of user-inputted review
user_query = input("\nEnter a review to predict its sentiment, or enter nothing to exit the program:\n")

while(user_query != ""):
    prediction = predict_sentiment(user_query)

    sentiment = interpret_prediction(prediction)

    print("\nSentiment: {} (Value: {})".format(sentiment, prediction))

    user_query = input("\n\nEnter a review to predict its sentiment, or enter nothing to exit the program:\n")