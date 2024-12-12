import pandas as pd
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Flatten, Concatenate, Input, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load dataset
places_df = pd.read_csv('dataset/tourismkelana_fixed.csv')

places_df.head()

# **Step 1: Price Categorization**
def categorize_price(df):
    conditions = [
        (df['Price'] < 50000),
        (df['Price'] >= 50000) & (df['Price'] <= 200000),
        (df['Price'] > 200000)
    ]
    categories = ['Murah', 'Sedang', 'Mahal']
    df['Price_Category'] = np.select(conditions, categories, default='Sedang')
    return df

places_df = categorize_price(places_df)

# **Step 2: Preprocessing Text**
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(nltk.corpus.stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the 'Category' and 'Description' columns
places_df['content'] = places_df['Category'] + ' ' + places_df['Description']
places_df['processed_content'] = places_df['content'].apply(preprocess_text)

# **Step 3: TF-IDF Vectorization**
tfidf_vectorizer = TfidfVectorizer(max_features=500)
tfidf_matrix = tfidf_vectorizer.fit_transform(places_df['processed_content']).toarray()

# **Step 4: Encoding Kategori dan Harga**
label_encoder_category = LabelEncoder()
places_df['Category_encoded'] = label_encoder_category.fit_transform(places_df['Category'])

label_encoder_price = LabelEncoder()
places_df['Price_Category_encoded'] = label_encoder_price.fit_transform(places_df['Price_Category'])

# Step 5: Create Labels Based on Rating (e.g., popular places)
places_df['Label'] = np.where(places_df['Rating'] >= 4.2, 1, 0)

places_df['Rating'].value_counts()

places_df.head()

# Step 6: Model Definition
# def build_model(input_dim_category, input_dim_price, input_dim_description, output_dim=128, dropout_rate=0.1):
#     # Category input and embedding
#     category_input = Input(shape=(1,), dtype=tf.int32, name="category")
#     category_embedding = Embedding(input_dim=input_dim_category, output_dim=output_dim,
#                                     embeddings_regularizer=l2(0.01))(category_input)
#     category_embedding = Flatten()(category_embedding)

#     # Price input and embedding
#     price_input = Input(shape=(1,), dtype=tf.int32, name="price")
#     price_embedding = Embedding(input_dim=input_dim_price, output_dim=output_dim,
#                                  embeddings_regularizer=l2(0.01))(price_input)
#     price_embedding = Flatten()(price_embedding)

#     # Description input (TF-IDF)
#     description_input = Input(shape=(input_dim_description,), dtype=tf.float32, name="description")

#     # Concatenate embeddings and TF-IDF input
#     x = Concatenate()([category_embedding, price_embedding, description_input])
#     x = Dense(32, activation='relu')(x)
#     x = Dropout(dropout_rate)(x)
#     x = Dense(16, activation='relu')(x)
#     x = Dropout(dropout_rate)(x)
#     output = Dense(1, activation='sigmoid')(x)

#     # Build and compile model
#     model = Model(inputs=[category_input, price_input, description_input], outputs=output)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

model = tf.keras.models.load_model('cbf_model.h5')

# Step 7: Prepare Data for K-Fold Cross-Validation
# category_input_data = places_df['Category_encoded'].values
# price_input_data = places_df['Price_Category_encoded'].values
# description_input_data = tfidf_matrix
# labels = places_df['Label'].values

# # Step 8: K-Fold Cross-Validation
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# val_losses = []
# val_accuracies = []

# for train_index, val_index in kf.split(category_input_data):
#     train_category, val_category = category_input_data[train_index], category_input_data[val_index]
#     train_price, val_price = price_input_data[train_index], price_input_data[val_index]
#     train_desc, val_desc = description_input_data[train_index], description_input_data[val_index]
#     train_labels, val_labels = labels[train_index], labels[val_index]

#     # Build the model
#     model = build_model(
#         input_dim_category=len(label_encoder_category.classes_),
#         input_dim_price=len(label_encoder_price.classes_),
#         input_dim_description=tfidf_matrix.shape[1],
#         dropout_rate=0.5  # Regularize with Dropout
#     )

#     # Train the model with Early Stopping
#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#     history = model.fit(
#         [train_category, train_price, train_desc],  # Training data
#         train_labels,                              # Training labels
#         validation_data=([val_category, val_price, val_desc], val_labels),  # Validation data
#         epochs=25,
#         batch_size=64,
#         callbacks=[early_stopping]
#     )

#     # Step 9: Evaluate Model
#     val_loss, val_accuracy = model.evaluate([val_category, val_price, val_desc], val_labels)
#     val_losses.append(val_loss)
#     val_accuracies.append(val_accuracy)

# # Step 10: Print Average Validation Loss and Accuracy
# print(f"Average Validation Loss: {np.mean(val_losses)}, Average Validation Accuracy: {np.mean(val_accuracies)}")

# # Step 11: Visualize Training History (Accuracy and Loss)
# # Note: You may want to modify this part to visualize each fold's history if needed
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.title('Model Accuracy and Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy / Loss')
# plt.show()

from sklearn.utils import shuffle

# **Step 6: Rekomendasi Berdasarkan Kota dan Harga**
def recommend(city_name, price_category, waktu, top_n=3):
    """
    Memberikan rekomendasi tempat wisata berdasarkan kota dan kategori harga.

    Parameters:
        city_name: str - Nama kota yang dipilih.
        price_category: str - Kategori harga yang diinginkan ('Murah', 'Sedang', 'Mahal').
        top_n: int - Jumlah rekomendasi yang diinginkan.

    Returns:
        pd.DataFrame - DataFrame yang berisi nama tempat, kategori, deskripsi, rating, dan harga.
    """
    # Filter berdasarkan kota dan harga
    tm = 0
    if waktu == "morning":
      tm = 10
    elif waktu == "afternoon":
      tm = 15
    elif waktu == "evening":
      tm = 21
    city_filtered_df = places_df[places_df['City'].str.contains(city_name, case=False, na=False)]
    time_filtered_df = city_filtered_df[city_filtered_df['Opening_Time'] <= tm]
    time_filtered_df = time_filtered_df[time_filtered_df['Closing_Time'] >= tm]
    price_filtered_df = time_filtered_df[time_filtered_df['Price_Category'] == price_category]

    if price_filtered_df.empty:
        return f"No places found in city '{city_name}' with price category '{price_category}'."

    # Prepare input data
    category_input_data = price_filtered_df['Category_encoded'].values.reshape(-1, 1)
    price_input_data = price_filtered_df['Price_Category_encoded'].values.reshape(-1, 1)
    description_input_data = tfidf_matrix[price_filtered_df.index]
   
    # Shuffle the input data to avoid bias in the recommendation
    category_input_data, price_input_data, description_input_data = shuffle(category_input_data, price_input_data, description_input_data)

    # Predict similarity
    predictions = model.predict([category_input_data, price_input_data, description_input_data]).flatten()

    # Get top_n recommendations
    similar_indices = predictions.argsort()[-top_n:][::-1]
    recommended_places = price_filtered_df.iloc[similar_indices]

    # Output recommendations
    return recommended_places[['Place_Name', 'Category', 'Description', 'Rating', 'Price', 'Lat', 'Long', 'Opening_Time', 'Closing_Time']]

recommendations = recommend("Surabaya", "Murah", "Evening", 50)
print(len(recommendations))
print(recommendations)

# Menyimpan model CBF yang sudah dilatih ke dalam file H5
# model.save('cbf_model.h5')

