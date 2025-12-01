import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import pickle
import os
import json

tf.random.set_seed(42)
np.random.seed(42)

def load_real_data(flood_file_path, precip_file_path):
    try:
        flood_df = pd.read_excel(flood_file_path)
        precip_df = pd.read_excel(precip_file_path)
        
        # NOTE: Ensure your Excel file headers match these English keys
        flood_feature_start_col = "Total_Water_Resources" 
        flood_feature_cols = []
        found_start = False
        
        for col in flood_df.columns:
            if col == flood_feature_start_col:
                found_start = True
            if found_start:
                flood_feature_cols.append(col)
        
        flood_features = flood_df[['Year', 'City'] + flood_feature_cols].copy()
        
        precip_feature_cols = precip_df.columns[2:].tolist()
        precip_features = precip_df[['city', 'year'] + precip_feature_cols].copy()
        
        combined_data = pd.merge(
            flood_features, 
            precip_features, 
            left_on=['City', 'Year'], 
            right_on=['city', 'year'], 
            how='inner'
        )
        
        flood_feature_data = combined_data[flood_feature_cols].copy()
        precip_feature_data = combined_data[precip_feature_cols].copy()
        
        flood_feature_data = flood_feature_data.fillna(flood_feature_data.median())
        precip_feature_data = precip_feature_data.fillna(precip_feature_data.median())
        
        flood_feature_data = flood_feature_data.replace([np.inf, -np.inf], np.nan)
        precip_feature_data = precip_feature_data.replace([np.inf, -np.inf], np.nan)
        
        flood_feature_data = flood_feature_data.fillna(flood_feature_data.median())
        precip_feature_data = precip_feature_data.fillna(precip_feature_data.median())
        
        flood_scaler = StandardScaler()
        precip_scaler = StandardScaler()
        
        flood_matrix = flood_scaler.fit_transform(flood_feature_data)
        precip_matrix = precip_scaler.fit_transform(precip_feature_data)
        
        return {
            'flood_matrix': flood_matrix.astype(np.float32),
            'precip_matrix': precip_matrix.astype(np.float32),
            'cities': combined_data['City'].values,
            'years': combined_data['Year'].values,
            'flood_feature_names': flood_feature_cols,
            'precip_feature_names': precip_feature_cols,
            'flood_scaler': flood_scaler,
            'precip_scaler': precip_scaler,
            'metadata': combined_data[['City', 'Year', 'city', 'year']],
            'flood_feature_data': flood_feature_data,
            'precip_feature_data': precip_feature_data,
            'combined_data': combined_data
        }
        
    except Exception:
        return None

def save_training_results(model, history, precip_encoder, flood_encoder, precip_autoencoder, flood_autoencoder, data, save_dir='saved_model'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    try:
        precip_encoder.save(os.path.join(save_dir, 'precip_encoder.h5'))
        flood_encoder.save(os.path.join(save_dir, 'flood_encoder.h5'))
        precip_autoencoder.save(os.path.join(save_dir, 'precip_autoencoder.h5'))
        flood_autoencoder.save(os.path.join(save_dir, 'flood_autoencoder.h5'))
        
        model.save_weights(os.path.join(save_dir, 'dual_autoencoder_weights.h5'))
        
        with open(os.path.join(save_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        preprocessing_info = {
            'flood_feature_names': data['flood_feature_names'],
            'precip_feature_names': data['precip_feature_names'],
            'cities': data['cities'].tolist(),
            'years': data['years'].tolist(),
            'flood_scaler': data['flood_scaler'],
            'precip_scaler': data['precip_scaler'],
            'flood_matrix_shape': data['flood_matrix'].shape,
            'precip_matrix_shape': data['precip_matrix'].shape
        }
        
        with open(os.path.join(save_dir, 'preprocessing_info.pkl'), 'wb') as f:
            pickle.dump(preprocessing_info, f)
        
        training_params = {
            'embedding_dim': 32,
            'epochs': 300,
            'batch_size': 32,
            'contrastive_weight': 3.0
        }
        
        with open(os.path.join(save_dir, 'training_params.json'), 'w') as f:
            json.dump(training_params, f, indent=4)
        
        model_info = {
            'precip_input_dim': data['precip_matrix'].shape[1],
            'flood_input_dim': data['flood_matrix'].shape[1],
            'embedding_dim': 32,
            'hidden_dims': [128, 64]
        }
        
        with open(os.path.join(save_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=4)
            
    except Exception:
        pass

def load_training_results(save_dir='saved_model'):
    if not os.path.exists(save_dir):
        return None
    
    try:
        with open(os.path.join(save_dir, 'model_info.json'), 'r') as f:
            model_info = json.load(f)
        
        precip_autoencoder, precip_encoder = create_autoencoder(
            model_info['precip_input_dim'], 
            model_info['embedding_dim'],
            model_info['hidden_dims']
        )
        flood_autoencoder, flood_encoder = create_autoencoder(
            model_info['flood_input_dim'], 
            model_info['embedding_dim'],
            model_info['hidden_dims']
        )
        
        model = ImprovedDualAutoencoderModel(
            precip_autoencoder, flood_autoencoder, precip_encoder, flood_encoder,
            contrastive_weight=3.0, temperature=0.1
        )
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
        model.load_weights(os.path.join(save_dir, 'dual_autoencoder_weights.h5'))
        
        precip_encoder = keras.models.load_model(os.path.join(save_dir, 'precip_encoder.h5'))
        flood_encoder = keras.models.load_model(os.path.join(save_dir, 'flood_encoder.h5'))
        precip_autoencoder = keras.models.load_model(os.path.join(save_dir, 'precip_autoencoder.h5'))
        flood_autoencoder = keras.models.load_model(os.path.join(save_dir, 'flood_autoencoder.h5'))
        
        with open(os.path.join(save_dir, 'training_history.pkl'), 'rb') as f:
            history = pickle.load(f)
        
        with open(os.path.join(save_dir, 'preprocessing_info.pkl'), 'rb') as f:
            preprocessing_info = pickle.load(f)
        
        return model, history, precip_encoder, flood_encoder, precip_autoencoder, flood_autoencoder, preprocessing_info
    
    except Exception:
        return None

def create_autoencoder(input_dim, embedding_dim, hidden_dims=[128, 64]):
    encoder_input = layers.Input(shape=(input_dim,))
    x = encoder_input
    for hidden_dim in hidden_dims:
        x = layers.Dense(hidden_dim, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    encoded = layers.Dense(embedding_dim, activation='relu', name='embedding')(x)
    
    x = encoded
    for hidden_dim in reversed(hidden_dims):
        x = layers.Dense(hidden_dim, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    decoded = layers.Dense(input_dim, activation='linear')(x)
    
    autoencoder = keras.Model(encoder_input, decoded)
    encoder = keras.Model(encoder_input, encoded)
    
    return autoencoder, encoder

class ImprovedDualAutoencoderModel(keras.Model):
    def __init__(self, precip_autoencoder, flood_autoencoder, precip_encoder, flood_encoder, 
                 contrastive_weight=2.0, temperature=0.1):
        super(ImprovedDualAutoencoderModel, self).__init__()
        self.precip_autoencoder = precip_autoencoder
        self.flood_autoencoder = flood_autoencoder
        self.precip_encoder = precip_encoder
        self.flood_encoder = flood_encoder
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

class DualInputDataGenerator(keras.utils.Sequence):
    def __init__(self, precip_data, flood_data, batch_size=32, shuffle=True):
        self.precip_data = precip_data
        self.flood_data = flood_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(precip_data))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.precip_data) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.precip_data))
        
        batch_precip = self.precip_data[start_idx:end_idx]
        batch_flood = self.flood_data[start_idx:end_idx]
        
        return (batch_precip, batch_flood)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            self.precip_data = self.precip_data[self.indexes]
            self.flood_data = self.flood_data[self.indexes]

def train_improved_model(data, embedding_dim=32, epochs=300, batch_size=32, contrastive_weight=3.0):
    precip_matrix = data['precip_matrix']
    flood_matrix = data['flood_matrix']
    
    precip_autoencoder, precip_encoder = create_autoencoder(
        precip_matrix.shape[1], embedding_dim)
    flood_autoencoder, flood_encoder = create_autoencoder(
        flood_matrix.shape[1], embedding_dim)
    
    model = ImprovedDualAutoencoderModel(
        precip_autoencoder, flood_autoencoder, precip_encoder, flood_encoder,
        contrastive_weight=contrastive_weight, temperature=0.1
    )
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    
    data_generator = DualInputDataGenerator(precip_matrix, flood_matrix, batch_size=batch_size)
    
    history = {
        'total_loss': [],
        'recon_loss': [],
        'contrastive_loss': []
    }
    
    for epoch in range(epochs):
        epoch_total_loss = []
        epoch_recon_loss = []
        epoch_contrastive_loss = []
        
        for batch_idx in range(len(data_generator)):
            batch_data = data_generator[batch_idx]
            
            with tf.GradientTape() as tape:
                precip_data, flood_data = batch_data
                
                precip_recon = precip_autoencoder(precip_data)
                flood_recon = flood_autoencoder(flood_data)
                
                recon_loss = (tf.reduce_mean(tf.square(precip_data - precip_recon)) + 
                             tf.reduce_mean(tf.square(flood_data - flood_recon))) / 2
                
                precip_embeddings = precip_encoder(precip_data)
                flood_embeddings = flood_encoder(flood_data)
                
                batch_size_tf = tf.shape(precip_embeddings)[0]
                precip_embeddings = tf.math.l2_normalize(precip_embeddings, axis=1)
                flood_embeddings = tf.math.l2_normalize(flood_embeddings, axis=1)
                
                similarity_matrix = tf.matmul(precip_embeddings, flood_embeddings, transpose_b=True) / 0.1
                labels = tf.range(batch_size_tf)
                
                precip_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=similarity_matrix)
                flood_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=tf.transpose(similarity_matrix))
                
                contrastive_loss = (tf.reduce_mean(precip_loss) + tf.reduce_mean(flood_loss)) / 2
                
                total_loss = recon_loss + contrastive_weight * contrastive_loss
            
            gradients = tape.gradient(total_loss, 
                                    precip_autoencoder.trainable_variables + 
                                    flood_autoencoder.trainable_variables)
            model.optimizer.apply_gradients(
                zip(gradients, 
                    precip_autoencoder.trainable_variables + 
                    flood_autoencoder.trainable_variables))
            
            epoch_total_loss.append(total_loss.numpy())
            epoch_recon_loss.append(recon_loss.numpy())
            epoch_contrastive_loss.append(contrastive_loss.numpy())
        
        avg_total_loss = np.mean(epoch_total_loss)
        avg_recon_loss = np.mean(epoch_recon_loss)
        avg_contrastive_loss = np.mean(epoch_contrastive_loss)
        
        history['total_loss'].append(avg_total_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['contrastive_loss'].append(avg_contrastive_loss)
        
        if epoch > 50 and avg_total_loss > np.min(history['total_loss'][-50:]):
            break
    
    return model, history, precip_encoder, flood_encoder, precip_autoencoder, flood_autoencoder

def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['total_loss'], label='Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['recon_loss'], label='Reconstruction Loss', color='orange')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['contrastive_loss'], label='Contrastive Loss', color='green')
    plt.title('Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')

def analyze_city_year_correlation(data, precip_reconstructed, flood_reconstructed):
    n_samples = len(data['precip_matrix'])
    cities = data['cities']
    years = data['years']
    
    sample_correlations = []
    
    for i in range(n_samples):
        precip_original_sample = data['precip_matrix'][i]
        precip_reconstructed_sample = precip_reconstructed[i]
        flood_original_sample = data['flood_matrix'][i]
        flood_reconstructed_sample = flood_reconstructed[i]
        
        precip_corr = np.corrcoef(precip_original_sample, precip_reconstructed_sample)[0, 1]
        flood_corr = np.corrcoef(flood_original_sample, flood_reconstructed_sample)[0, 1]
        overall_corr = (precip_corr + flood_corr) / 2
        
        sample_correlations.append({
            'city': cities[i],
            'year': years[i],
            'precip_correlation': precip_corr,
            'flood_correlation': flood_corr,
            'overall_correlation': overall_corr
        })
    
    correlation_df = pd.DataFrame(sample_correlations)
    
    city_correlations = correlation_df.groupby('city').agg({
        'precip_correlation': ['mean', 'std', 'count'],
        'flood_correlation': ['mean', 'std', 'count'],
        'overall_correlation': ['mean', 'std', 'count']
    }).round(4)
    
    city_correlations.columns = ['precip_mean', 'precip_std', 'precip_count',
                                'flood_mean', 'flood_std', 'flood_count',
                                'overall_mean', 'overall_std', 'overall_count']
    
    city_correlations = city_correlations.sort_values('overall_mean', ascending=False)
    
    year_correlations = correlation_df.groupby('year').agg({
        'precip_correlation': ['mean', 'std', 'count'],
        'flood_correlation': ['mean', 'std', 'count'],
        'overall_correlation': ['mean', 'std', 'count']
    }).round(4)
    
    year_correlations.columns = ['precip_mean', 'precip_std', 'precip_count',
                                'flood_mean', 'flood_std', 'flood_count',
                                'overall_mean', 'overall_std', 'overall_count']
    
    year_correlations = year_correlations.sort_values('year')
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(correlation_df['overall_correlation'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(correlation_df['overall_correlation'].mean(), color='red', linestyle='--', 
                label=f'Mean: {correlation_df["overall_correlation"].mean():.4f}')
    plt.xlabel('Sample Overall Correlation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    top_cities = city_correlations.head(15)
    plt.bar(range(len(top_cities)), top_cities['overall_mean'], color='lightgreen', alpha=0.7)
    plt.xticks(range(len(top_cities)), top_cities.index, rotation=45)
    plt.xlabel('City')
    plt.ylabel('Mean Correlation')
    plt.title('Top 15 Cities by Reconstruction Correlation')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(year_correlations.index, year_correlations['overall_mean'], marker='o', linewidth=2, label='Overall')
    plt.plot(year_correlations.index, year_correlations['precip_mean'], marker='s', linewidth=2, label='Precipitation')
    plt.plot(year_correlations.index, year_correlations['flood_mean'], marker='^', linewidth=2, label='Flood')
    plt.xlabel('Year')
    plt.ylabel('Correlation')
    plt.title('Correlation Trends over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    plt.scatter(correlation_df['precip_correlation'], correlation_df['flood_correlation'], 
               alpha=0.6, c=correlation_df['overall_correlation'], cmap='viridis')
    plt.xlabel('Precipitation Correlation')
    plt.ylabel('Flood Correlation')
    plt.title('Precipitation vs Flood Correlation')
    plt.colorbar(label='Overall Correlation')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    best_samples = correlation_df.nlargest(5, 'overall_correlation')
    worst_samples = correlation_df.nsmallest(5, 'overall_correlation')
    
    plt.bar(range(5), best_samples['overall_correlation'], color='green', alpha=0.7, label='Best Samples')
    plt.bar(range(5, 10), worst_samples['overall_correlation'], color='red', alpha=0.7, label='Worst Samples')
    
    labels = [f"{city}\n{year}" for city, year in zip(best_samples['city'], best_samples['year'])] + \
             [f"{city}\n{year}" for city, year in zip(worst_samples['city'], worst_samples['year'])]
    
    plt.xticks(range(10), labels, rotation=45)
    plt.ylabel('Overall Correlation')
    plt.title('Best and Worst Reconstruction Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    selected_cities = city_correlations.head(10).index
    selected_data = correlation_df[correlation_df['city'].isin(selected_cities)]
    
    heatmap_data = selected_data.pivot_table(values='overall_correlation', 
                                           index='city', columns='year', 
                                           aggfunc='mean')
    
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.3f', cbar=True)
    plt.title('City-Year Correlation Heatmap')
    plt.xlabel('Year')
    plt.ylabel('City')
    
    plt.tight_layout()
    plt.savefig('city_year_correlation_analysis.png', dpi=300, bbox_inches='tight')
    
    correlation_df.to_csv('city_year_correlation_results.csv', index=False, encoding='utf-8-sig')
    city_correlations.to_csv('city_aggregated_correlations.csv', encoding='utf-8-sig')
    year_correlations.to_csv('year_aggregated_correlations.csv', encoding='utf-8-sig')
    
    return correlation_df, city_correlations, year_correlations

def main():
    # NOTE: Update these paths to your relative data paths
    flood_file_path = "./data/flooding_control.xlsx"
    precip_file_path = "./data/precipitation.xlsx"
    save_dir = 'saved_model'
    
    use_saved_model = False
    
    if use_saved_model:
        loaded_results = load_training_results(save_dir)
        if loaded_results is not None:
            model, history, precip_encoder, flood_encoder, precip_autoencoder, flood_autoencoder, preprocessing_info = loaded_results
            data = load_real_data(flood_file_path, precip_file_path)
        else:
            use_saved_model = False
    
    if not use_saved_model:
        data = load_real_data(flood_file_path, precip_file_path)
        if data is None:
            return
        
        model, history, precip_encoder, flood_encoder, precip_autoencoder, flood_autoencoder = train_improved_model(
            data, 
            embedding_dim=32, 
            epochs=300,
            batch_size=32, 
            contrastive_weight=3.0
        )
        
        save_training_results(model, history, precip_encoder, flood_encoder, precip_autoencoder, flood_autoencoder, data, save_dir)
    
    plot_training_history(history)
    
    precip_reconstructed = precip_autoencoder.predict(data['precip_matrix'], verbose=0)
    flood_reconstructed = flood_autoencoder.predict(data['flood_matrix'], verbose=0)
    
    analyze_city_year_correlation(data, precip_reconstructed, flood_reconstructed)

if __name__ == "__main__":
    main()