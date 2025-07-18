import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from utils import *
from model import create_metg_model
import time
import yaml
import joblib
from numpy.lib.stride_tricks import sliding_window_view
import polars as pl
import json
from scipy.stats.mstats import winsorize


# Programme d'entraînement du modèle METG pour une recette entrée
# Le programme va 1) concaténer l'entièreté des données correspondant à {recette} si elle est correcte, (ou si la concaténation existe déjà, passer direcetement à 2))
# 2)scaler et augmenter les données d'entraînement, puis entraîner le modèle sur ces données, puis enregistrer le scaler ainsi que le modèle pour réutilisation

def augment_time_series(windows, noise_std=0.01, scale_range=(0.9, 1.1), p_noise=0.5, p_scale=0.5):
    # Augmente une fenêtre entrée avec bruit gaussien + random scaling  (x0.9-x1.1)
    augmented = windows.copy()
    noise_mask = np.random.rand(len(windows)) < p_noise
    augmented[noise_mask] += np.random.normal(0, noise_std, size=augmented[noise_mask].shape)
    scale_mask = np.random.rand(len(windows)) < p_scale
    scales = np.random.uniform(*scale_range, size=(scale_mask.sum(), 1, 1))
    augmented[scale_mask] *= scales
    return augmented

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def concatenate_file(recette, columns_to_drop, bottom):

    # Pour une recette donnée, concaténe tous les fichiers présents dans le dossier cycles_merged
    output_dir = "output\\concatenated_files"
    os.makedirs(output_dir, exist_ok=True)
    merged_file = os.path.join(output_dir, f"all_{recette}.xlsx")
    dropped_info_file = os.path.join(output_dir, f"dropped_info_{recette}.yaml")

    if os.path.exists(merged_file) and os.path.getsize(merged_file) > 0:
        try:
            pl_merged = pl.read_excel(merged_file)
            merged = pl_merged.to_pandas()
            with open(dropped_info_file, "r", encoding="utf-8") as f:
                dropped_info = yaml.safe_load(f)
            final_columns = dropped_info["final_columns"]
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier Excel existant : {e}")
            print("Suppression du fichier corrompu et régénération...")
            os.remove(merged_file)
            if os.path.exists(dropped_info_file):
                os.remove(dropped_info_file)
            return concatenate_file(recette, columns_to_drop, bottom)
    else:
        print("Fichier concaténé non trouvé, concaténation des fichiers de la recette en cours...")
        data_dir = r"C:\Users\S643771\Documents\Programmes\donnees_machine\cycles_merged"
        file_list = []
        
        if bottom:
            # filtrage pour les fichiers bottom 80% anomaly scores
            chemin_fichier = f"output//bottom_80//{recette}.json"
            with open(chemin_fichier, 'r') as file:
                liste_fichiers_bottom = json.load(file)

            # Filtrage et tri par OS croissant

            all_fnames = [
                fname for fname in os.listdir(data_dir)
                if fname in liste_fichiers_bottom and fname.endswith(".xlsx") and len(fname.split("_")) >= 3 and fname.split("_")[1] == recette
            ]
        else:
            all_fnames = [
                fname for fname in os.listdir(data_dir)
                if fname.endswith(".xlsx") and len(fname.split("_")) >= 3 and fname.split("_")[1] == recette
            ]
        all_fnames = sorted(all_fnames, key=lambda fname: int(fname.split("_")[2]))
        total_files = len(all_fnames)
        for idx, fname in enumerate(all_fnames):

            # Affichage progression tous les 10%
            if total_files >= 10 and idx % max(1, total_files // 10) == 0 and idx != 0:
                percent = int((idx / total_files) * 100)
                print(f"Progression: {percent}% ({idx}/{total_files})")
            pl_df = pl.read_excel(os.path.join(data_dir, fname))
            df = pl_df.to_pandas()
            df = df.select_dtypes(include=[np.number])

            # Drop des colonnes spécifiées dans la config.yaml
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
            # Ajout du nom de fichier comme colonne (tracabilité)
            df["source_filename"] = fname
            file_list.append(df)
        if total_files % 2 != 0:
            print(f"Progression: 100% ({total_files}/{total_files})")

        if file_list:

            # Drop des colonnes présentes dans une minorité des fichiers

            all_columns = [set(df.columns) for df in file_list]
            common_columns = set.intersection(*all_columns)
            potential_columns_to_drop = set()

            for col_set in all_columns:
                potential_columns_to_drop.update(col_set - common_columns)

            minority_columns = []

            for col in potential_columns_to_drop:

                present_count = sum(1 for df in file_list if col in df.columns)
                absent_count = len(file_list) - present_count
                print(f"\nColonne '{col}' présente dans {present_count} fichier(s) et absente dans {absent_count} fichier(s).")

                if present_count < absent_count:
                    minority_columns.append(col)

            # Drop des colonnes minoritaires et des colonnes vides
            for df in file_list:
                df.drop(columns=minority_columns, inplace=True, errors='ignore')
                df.dropna(axis=1, how='all', inplace=True)

            merged = pd.concat(file_list, ignore_index=True)
            merged = merged.reindex(columns=sorted(list(set.union(*[set(df.columns) for df in file_list]))))

            # Suppression des lignes contenant des zéros / valeurs manquantes
            # Ne pas prendre en compte la colonne 'source_filename' pour ce filtrage
            feature_cols = [col for col in merged.columns if col != "source_filename"]
            # More lenient filtering - don't remove rows with zeros in engineered features
            original_features = [col for col in feature_cols if not any(
                engineered in col for engineered in ['rolling', 'trend', 'diff', 'percentile', 'zscore', 'anomaly_score']
            )]
            merged = merged[(merged[original_features] != 0).all(axis=1)]
            merged = merged.dropna(axis=0, how='any', subset=feature_cols)
            
            merged["Échelles"] = winsorize(merged["Échelles"], limits = [0.01, 0.01])
            merged["Disque"] = winsorize(merged["Disque"], limits = [0.01, 0.01])

            final_columns = [col for col in merged.columns if col != "source_filename"]
            print(f"Colonnes finales utilisées pour l'entraînement : {final_columns}")

            # Sauvegarde des colonnes finales, colonnes droppées et colonnes minoritaires
            dropped_info = {
                "final_columns": final_columns,
                "minority_columns": minority_columns,
                "columns_to_drop": columns_to_drop
            }

            with open(dropped_info_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(dropped_info, f)

            try:
                print("\nEnregistrement du fichier concaténé...")
                merged.to_excel(merged_file, index=False)
                print(f"Fichier concaténé sauvegardé: {merged_file}")

            except Exception as e:
                print(f"Erreur lors de l'enregistrement du fichier concaténé: {e}")
                sys.exit(1)
        else:
            print("Fichiers valides introuvables.")
            sys.exit(1)

    print(f"\nDataset chargé. Shape: {merged.shape}")
    return merged

def inversion_debit_deau(x):
    return 15 - x

def creation_scaler(merged):
    scalers = {}
    scaled_data = merged.copy()
    
    # Save winsorization parameters for Échelles and Disque
    echelles_limits = {}
    disque_limits = {}
    
    if "Échelles" in merged.columns:
        # Calculate percentiles before winsorization
        echelles_q01 = merged["Échelles"].quantile(0.01)
        echelles_q99 = merged["Échelles"].quantile(0.99)
        echelles_limits = {"lower": echelles_q01, "upper": echelles_q99}
        scaled_data["Échelles"] = winsorize(scaled_data["Échelles"], limits=[0.01, 0.01])
    
    if "Disque" in merged.columns:
        disque_q01 = merged["Disque"].quantile(0.01)
        disque_q99 = merged["Disque"].quantile(0.99)
        disque_limits = {"lower": disque_q01, "upper": disque_q99}
        scaled_data["Disque"] = winsorize(scaled_data["Disque"], limits=[0.01, 0.01])
    
    # Create scalers for each column
    for col in scaled_data.columns:
        if col == "Débit deau [l/min]":
            # Inversion transformation
            invert_transformer = FunctionTransformer(inversion_debit_deau, validate=False)
            inverted_values = invert_transformer.fit_transform(scaled_data[[col]])
            
            minmax_scaler = MinMaxScaler()
            minmax_scaler.fit(inverted_values)
            
            scalers[col] = {
                'invert': invert_transformer,
                'minmax': minmax_scaler
            }
        else:
            minmax_scaler = MinMaxScaler()
            minmax_scaler.fit(scaled_data[[col]])
            scalers[col] = {'minmax': minmax_scaler}
    
    scalers['_winsorize_limits'] = {
        'Échelles': echelles_limits,
        'Disque': disque_limits
    }
    
    return scaled_data, scalers

def train(merged, recette, window_size, BATCH_SIZE, N_EPOCHS, hidden_size, d_model, memory_size, nhead, num_layers, k_neighbors, temperature, learning_rate, lambda1, lambda2):
    merged.drop(columns=["source_filename"], inplace=True, errors="ignore")
    print(f"\nTRAINING pour la RECETTE {recette} avec METG")
    
    scalers_file = f"output//models//{recette}_metg_scalers.pkl"
    
    try:
        if os.path.exists(scalers_file):
            print(f"Scalers trouvés pour la recette {recette}, chargement...")
            scalers = joblib.load(scalers_file)
            
            merged_scaled = merged.copy()
            
            winsorize_limits = scalers.get('_winsorize_limits', {})
            
            if "Échelles" in merged_scaled.columns and 'Échelles' in winsorize_limits:
                limits = winsorize_limits['Échelles']
                merged_scaled["Échelles"] = merged_scaled["Échelles"].clip(lower=limits['lower'], upper=limits['upper'])
            
            if "Disque" in merged_scaled.columns and 'Disque' in winsorize_limits:
                limits = winsorize_limits['Disque']
                merged_scaled["Disque"] = merged_scaled["Disque"].clip(lower=limits['lower'], upper=limits['upper'])
            
            for col in merged_scaled.columns:
                if col in scalers:
                    if col == "Débit deau [l/min]":
                        invert = scalers[col]['invert']
                        minmax = scalers[col]['minmax']
                        vals = invert.transform(merged_scaled[[col]])
                        merged_scaled[col] = minmax.transform(vals).flatten()
                    else:
                        minmax = scalers[col]['minmax']
                        merged_scaled[col] = minmax.transform(merged_scaled[[col]]).flatten()
            
            print("Scalers existants utilisés avec succès")
        else:
            print("Scalers non trouvés, création de nouveaux scalers...")
            merged_scaled, scalers = creation_scaler(merged)
            
            os.makedirs("output//models", exist_ok=True)
            
            joblib.dump(scalers, scalers_file)
            print("Nouveaux scalers créés et enregistrés")

        x_scaled = merged_scaled.values

        windows = sliding_window_view(x_scaled, (window_size, x_scaled.shape[1]))
        windows = windows.reshape(-1, window_size, merged_scaled.shape[1])
        print(f"Windows shape: {windows.shape}")

        # Get input dimensions
        input_dim = windows.shape[2]  # Number of features
        seq_len = windows.shape[1]    # Sequence length

        windows_train = windows[:int(np.floor(.8 * windows.shape[0]))]
        windows_val = windows[int(np.floor(.8 * windows.shape[0])):windows.shape[0]]

        # Augmentation du dataset d'entraînement
        windows_train_aug = augment_time_series(windows_train)
        print(f"Augmented train windows: {windows_train_aug.shape}, Val windows: {windows_val.shape}")

        # Convert to tensors - keep 3D shape for METG
        train_tensor = torch.from_numpy(windows_train_aug).float()
        val_tensor = torch.from_numpy(windows_val).float()
        
        train_loader = data_utils.DataLoader(data_utils.TensorDataset(train_tensor), 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True,  
                                           num_workers=4, 
                                           pin_memory=True if torch.cuda.is_available() else False)

        val_loader = data_utils.DataLoader(data_utils.TensorDataset(val_tensor), 
                                         batch_size=BATCH_SIZE, 
                                         shuffle=False, 
                                         num_workers=2,
                                         pin_memory=True if torch.cuda.is_available() else False)
        print(f"Train loader batches: {len(train_loader)}; Val loader batches: {len(val_loader)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDEVICE == {device}\n")

        # Create METG model
        model = create_metg_model(
            input_dim=input_dim,
            d_model=d_model,
            memory_size=memory_size,
            seq_len=seq_len,
            nhead=nhead,
            num_layers=num_layers,
            k_neighbors=k_neighbors,
            temperature=temperature
        )
        
        # Set loss weights
        model.lambda1 = lambda1
        model.lambda2 = lambda2
        
        model = to_device(model, device)

        # Set up optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        start = time.time()
        print("-> Début de l'entraînement METG...")
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(N_EPOCHS):
            # Training phase
            model.train()
            train_losses = []
            
            for batch_idx, (batch_data,) in enumerate(train_loader):
                batch_data = batch_data.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                transformer_output, gcn_output, attention_weights = model(batch_data)
                
                # Compute loss
                total_loss, loss_dict = model.compute_loss(
                    batch_data, transformer_output, gcn_output, attention_weights
                )
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                train_losses.append(total_loss.item())
            
            # Validation phase
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_data, in val_loader:
                    batch_data = batch_data.to(device)
                    
                    transformer_output, gcn_output, attention_weights = model(batch_data)
                    total_loss, loss_dict = model.compute_loss(
                        batch_data, transformer_output, gcn_output, attention_weights
                    )
                    
                    val_losses.append(total_loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
                print(f"Epoch {epoch+1:3d}/{N_EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Save model
        torch.save(model.state_dict(), f"output//models//{recette}_metg_model.pth")
        
        # Save model parameters for later loading
        model_params = {
            'input_dim': input_dim,
            'd_model': d_model,
            'memory_size': memory_size,
            'seq_len': seq_len,
            'nhead': nhead,
            'num_layers': num_layers,
            'k_neighbors': k_neighbors,
            'temperature': temperature,
            'lambda1': lambda1,
            'lambda2': lambda2
        }
        
        with open(f"output//models//{recette}_metg_params.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(model_params, f)

        print(f'Entraînement terminé, modèle METG avec les meilleurs pertes sauvegardé : "{recette}_metg_model.pth"')
        print(f"Temps total: {time.time() - start:.2f} secondes")
        return True
    except Exception as e:
        print(f"Erreur lors du training METG {e}")
        sys.exit(1)

if __name__ == "__main__":

    recettes = [sys.argv[1]]
    
    if recettes:
        config = load_config("config.yaml")
        columns_to_drop = config["columns_to_drop"]
        window_size = config["window_size"]
        BATCH_SIZE = config["batch_size"]
        N_EPOCHS = config["n_epochs"]
        hidden_size = config["hidden_size"]
        
        # METG specific parameters
        d_model = config["d_model"]
        memory_size = config["memory_size"]
        nhead = config["nhead"]
        num_layers = config["num_layers"]
        k_neighbors = config["k_neighbors"]
        temperature = config["temperature"]
        learning_rate = config["learning_rate"]
        lambda1 = config["lambda1"]
        lambda2 = config["lambda2"]

        for recette in ["36", "607", "1003", "1115", "1116", "1260", "1280", "1601", "1900", "1962", "2963"]:
        # for recette in recettes:
            merged = concatenate_file(recette, columns_to_drop, bottom=True)
            trained = train(merged, recette, window_size, BATCH_SIZE, N_EPOCHS, hidden_size, 
                          d_model, memory_size, nhead, num_layers, k_neighbors, temperature, 
                          learning_rate, lambda1, lambda2)
        
        # if trained:
        #     sys.exit(0)
        # else:
        #     sys.exit(1)
    else:
        print("Erreur, utilisation : python train_metg.py {recette}")