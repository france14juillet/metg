import os
import pandas as pd
import numpy as np
import torch
from usad import UsadModel, to_device, testing
import sys
import yaml

import plotly.graph_objects as go

from knee import compute_and_graph_treshold
import joblib
from utils import get_threshold
import polars as pl
from scipy.stats.mstats import winsorize

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Programme de calcul de scores d'anomalies pour chacune des fenêtres de chacun des cycles merged, ce programme est préléminaire à l'utilisation de temporal.py qui lui va tracer les évolutions des 
# anomaly scores de façon temporelle, avec limite calculée par knee.py et highlight des fichiers Non Conformes

def plot_anomaly_scores_with_threshold(input_xlsx, threshold):
    pl_df = pl.read_excel(input_xlsx)
    df = pl_df.to_pandas()

    df_sorted = df.copy()
    try:
        df_sorted["sort_key"] = df_sorted.iloc[:, 2].apply(lambda x: int(str(x).split("_")[2]))
        df_sorted = df_sorted.sort_values("sort_key").reset_index(drop=True)
    except Exception as e:
        print("Erreur lors du tri des fichiers :", e)
        df_sorted = df 

    anomaly_scores = df_sorted.iloc[:, 1].values
    filenames = df_sorted.iloc[:, 2].astype(str).values
    indices = np.arange(len(anomaly_scores))
    above = anomaly_scores > threshold
    starts_with_2 = np.array([f.startswith("2") for f in filenames])

    colors = []
    for i in range(len(anomaly_scores)):
        if starts_with_2[i] and above[i]:
            colors.append('green')
        elif starts_with_2[i]:
            colors.append('yellow')
        elif above[i]:
            colors.append('red')
        else:
            colors.append('blue')

    segments = []
    current_color = colors[0]
    seg_x = [indices[0]]
    seg_y = [anomaly_scores[0]]

    for i in range(1, len(anomaly_scores)):
        if colors[i] == current_color:
            seg_x.append(indices[i])
            seg_y.append(anomaly_scores[i])
        else:
            segments.append((seg_x, seg_y, current_color))
            seg_x = [indices[i-1], indices[i]]
            seg_y = [anomaly_scores[i-1], anomaly_scores[i]]
            current_color = colors[i]
    segments.append((seg_x, seg_y, current_color))

    fig = go.Figure()

    for seg_x, seg_y, color in segments:
        fig.add_trace(go.Scattergl(
            x=seg_x, y=seg_y,
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        ))

    fig.add_hline(y=threshold, line_dash="dash", line_color="blue", annotation_text=f"Threshold={threshold}")

    fig.add_trace(go.Scattergl(x=[None], y=[None], mode='lines', line=dict(color='red', width=2), name='> Threshold'))
    fig.add_trace(go.Scattergl(x=[None], y=[None], mode='lines', line=dict(color='blue', width=2), name='≤ Threshold'))
    fig.add_trace(go.Scattergl(x=[None], y=[None], mode='lines', line=dict(color='orange', width=2), name='Non conforme'))

    fig.update_layout(
        title=f"[{recette}] Anomaly Scores coloré par over/under threshold",
        xaxis_title="Indice fenêtre",
        yaxis_title="Anomaly Score",
        legend=dict(itemsizing='constant'),
        width=1200,
        height=500
    )
    fig.show()

def compute_thresholds(recette):
    input_xlsx = fr"C:\Users\S643771\Documents\Programmes\donnees_machine\usad-master\output\anomaly_scores\{recette}_anomaly_results.xlsx"
    threshold_dir = r"C:\Users\S643771\Documents\Programmes\donnees_machine\usad-master\output\threshold"
    if os.path.exists(input_xlsx):
        threshold = get_threshold(recette, threshold_dir)
        plot_anomaly_scores_with_threshold(input_xlsx, threshold)
    else:
        print("Fichier non existant input_xlsx")

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def inversion_debit_deau(x):
    return 15 - x

def transform_with_saved_scalers(df, scalers, columns):
    # Préprocessing similaire a celui utilisé pour l'entraînement
    df_out = df.copy()
    
    winsorize_limits = scalers.get('_winsorize_limits', {})
    
    for col in columns:
        if col in winsorize_limits and winsorize_limits[col]:
            limits = winsorize_limits[col]
            df_out[col] = df_out[col].clip(lower=limits['lower'], upper=limits['upper'])
        
        if col == "Débit deau [l/min]":
            invert = scalers[col]['invert']
            minmax = scalers[col]['minmax']
            vals = invert.transform(df_out[[col]])
            df_out[col] = minmax.transform(vals)
        else:
            minmax = scalers[col]['minmax']
            df_out[col] = minmax.transform(df_out[[col]])
    
    return df_out

def compute_anomaly(recette, window_size, hidden_size, columns_to_drop):
    try:
        print("Chargement des données...")
        train_df = pd.read_excel(f"output//concatenated_files//all_{recette}.xlsx")

        # Chargement fichier contenant les colonnes utilisées lors du training
        with open(f"output//concatenated_files//dropped_info_{recette}.yaml", "r", encoding="utf-8") as f:
            dropped_info = yaml.safe_load(f)
        final_columns = dropped_info["final_columns"]

        num_features = len(final_columns)
        w_size = window_size * num_features
        z_size = window_size * hidden_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Chargement du modèle pour la recette {recette}...")
        model = UsadModel(w_size, z_size)
        checkpoint = torch.load(f"output//models//{recette}_model.pth", map_location=device)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder1.load_state_dict(checkpoint['decoder1'])
        model.decoder2.load_state_dict(checkpoint['decoder2'])
        model = to_device(model, device)
        model.eval()

        print("Chargement des scalers...")
        scalers_path = f"output//models//{recette}_scalers.pkl"
        if not os.path.exists(scalers_path):
            raise FileNotFoundError(f"Scaler file not found: {scalers_path}")
        scalers = joblib.load(scalers_path)

        # Traitement des fichiers des données machine 
        data_dir = r"C:\Users\S643771\Documents\Programmes\donnees_machine\cycles_merged"
        output_dir = "output\\anomaly_scores"
        os.makedirs(output_dir, exist_ok=True)

        all_files = [
            fname for fname in os.listdir(data_dir)
            if fname.endswith(".xlsx") and len(fname.split("_")) >= 2 and fname.split("_")[1] == recette
        ]
        total_files = len(all_files)
        if total_files == 0:
            print("Aucun fichier à traiter.")
            return

        all_results = []
        skipped_files = []
        list_df_scaled = []
        
        for idx, fname in enumerate(all_files):
            # Print du progrès tous les 10%
            if total_files >= 10 and idx % max(1, total_files // 10) == 0 and idx != 0:
                percent = int((idx / total_files) * 100)
                print(f"Progression: {percent}% ({idx}/{total_files})")

            df = pd.read_excel(os.path.join(data_dir, fname))
            df = df.select_dtypes(include=[np.number])
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

            # Vérifier la présence de toutes les colonnes nécessaires
            missing_cols = [col for col in final_columns if col not in df.columns]
            if missing_cols:
                skipped_files.append((fname, missing_cols))
                continue

            # Garder uniquement les colonnes finales et supprimer les lignes contenant des zéros
            df = df[final_columns]
            df = df[(df != 0).all(axis=1)]
            df = df.dropna()

            if df.shape[0] < window_size:
                print(f"Fichier {fname} trop court après suppression des zéros : skipping.")
                continue

            df_scaled = transform_with_saved_scalers(df, scalers, final_columns)

        #     AFFICHAGE DONNEES SCALÉES

        #     list_df_scaled.append(df_scaled)

        # final_df_scaled = pd.concat(list_df_scaled)
        
        # n_cols = len(final_df_scaled.columns)

        # fig = make_subplots(rows=n_cols, cols=1,
        #                     subplot_titles=final_df_scaled.columns.tolist())

        # for i, col in enumerate(final_df_scaled.columns):
        #     fig.add_trace(
        #         go.Scattergl(x=final_df_scaled.index, y=final_df_scaled[col], name=col, mode='lines+markers'),
        #         row=i+1, col=1
        #     )

        # fig.show()



            windows = df_scaled.values[np.arange(window_size)[None, :] + np.arange(df_scaled.shape[0] - window_size)[:, None]]
            windows = windows.reshape(windows.shape[0], -1)
            windows_tensor = torch.from_numpy(windows).float()
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(windows_tensor),
                batch_size=256, shuffle=False
            )
            
            scores = testing(model, test_loader)
            scores = torch.cat(scores).cpu().numpy()
            
            out_df = pd.DataFrame({
                "window_start": np.arange(len(scores)),
                "anomaly_score": scores
            })
            out_df["filename"] = fname
            all_results.append(out_df)
        print(f"Progression: 100% ({len(all_files)}/{len(all_files)})")

        # Log des fichiers skippés
        if skipped_files:
            print("\nFichiers ignorés pour colonnes manquantes :")
            for fname, missing_cols in skipped_files:
                print(f"  {fname} : colonnes manquantes {missing_cols}")

        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            final_results.to_excel(os.path.join(output_dir, f"{recette}_anomaly_results.xlsx"), index=False)
            print(f"Scores d'anomalies enregistrés à '{recette}_anomaly_results.xlsx'")
        else:
            print("Aucun fichier valide.")
    
        print("Tous les fichiers ont été traité.")

        return True
    except Exception as e:
        print(f"Erreur computing {e}")
        sys.exit(1)

if __name__ == "__main__":
    config = load_config("config.yaml")
    window_size = config["window_size"]
    hidden_size = config["hidden_size"]
    columns_to_drop = config["columns_to_drop"]

    if len(sys.argv) > 1:

        if sys.argv[1] == "all":
            recette_entree = ["36", "607", "1003", "1115", "1116", "1260", "1280", "1601", "1900", "1962","2963"]
        else:
            print(sys.argv)
            recette_entree = sys.argv[1]
            print(recette_entree)
            
        for recette in recette_entree:
            computed = compute_anomaly(recette, window_size, hidden_size, columns_to_drop)
        
            if computed:
                print("Calcul et traçage de threshold d'anomaly value")
                compute_and_graph_treshold(recette, 3.0)
                # sys.exit(0)
            else:
                print("Erreur lors du calcul de la limite de score")
    else:
        print("Pas de recette entrée")
        print("Utilisation : python compute.py <recette>")
        sys.exit(1)