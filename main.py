import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("C:/Users/Harvey/Downloads/animelists_cleaned.csv")

df_num = df[['anime_id', 'my_watched_episodes', 'my_score', 'my_status',
       'my_rewatching', 'my_rewatching_ep']]

print("done with df num")
# --- 1. Data Preprocessing --- 
anime_data = pd.read_csv("C:/Users/Harvey/Downloads/anime-with-description.csv")

# Merge DataFrames
merged_df = pd.merge(df_num, anime_data, on='anime_id')

merged_df = merged_df.dropna(subset=['anime_id'])

merged_df = merged_df.dropna(subset=['synopsis'])

print("done with merging")


# Assuming your DataFrame is named merged_df
# Replace this with your actual merged_df loading if necessary

print("# 1. Preprocessing")
# 1. Preprocessing
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")  # Example: top 1000 words + out-of-vocab token
tokenizer.fit_on_texts(merged_df['synopsis'])
sequences = tokenizer.texts_to_sequences(merged_df['synopsis'])
padded_sequences = pad_sequences(sequences, padding='post')  # Pad sequences to the same length

print("# 2. Create TensorFlow Dataset")
# 2. Create TensorFlow Dataset
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, merged_df['anime_id'].values, test_size=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(2)  # Example batch size
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(2)

print("# 3. Build a Model")
# 3. Build a Model
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding (index 0)
embedding_dim = 16

model = tf.keras.Sequential([
    Embedding(vocab_size, embedding_dim, input_length=padded_sequences.shape[1]),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(merged_df['anime_id'].unique()))  # Output layer: predict anime IDs
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

print("Train the Model")
# 4. Train the Model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)  # Adjust epochs as needed

# 5. Make Recommendations
def get_recommendations(synopsis, top_n=5):
    sequence = tokenizer.texts_to_sequences([synopsis])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=padded_sequences.shape[1])
    prediction_vector = model.predict(padded_sequence)

    # Calculate cosine similarity with other anime embeddings (optional)
    all_anime_embeddings = model.predict(padded_sequences) 
    similarities = cosine_similarity(prediction_vector, all_anime_embeddings)
    similar_anime_indices = similarities.argsort()[0][::-1][:top_n]  # Get top N most similar
    
    recommended_anime_ids = merged_df['anime_id'].iloc[similar_anime_indices].tolist()
    return recommended_anime_ids

# Example usage
print("usage")
new_synopsis = "A robot discovers its humanity in a futuristic city."
recommendations = get_recommendations(new_synopsis)
print("Recommended anime IDs:", recommendations) 