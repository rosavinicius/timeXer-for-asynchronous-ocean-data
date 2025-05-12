# psod_dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset

def load_and_merge_parquet(files, key='timestamp', how='inner'):
    """
    files: lista com os caminhos dos arquivos Parquet.
    key: coluna comum para o merge.
    how: método de merge ('inner' ou 'outer').
    """
    dfs = [pd.read_parquet(f) for f in files]
    # Supondo que cada DataFrame contenha a coluna 'timestamp'
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on=key, how=how)
    # Ordena pelo timestamp, se necessário
    df_merged = df_merged.sort_values(by=key)
    return df_merged

class PSODDataset(Dataset):
    def __init__(self, dataframe, seq_len, target_col=ssh_praticagem):
        """
        dataframe: DataFrame combinado com todas as features.
        seq_len: Comprimento da sequência de entrada.
        target_col: Nome da coluna alvo, se for uma tarefa de previsão univariada.
                    Se None, assume que todas as colunas são usadas.
        """
        # Se não houver coluna alvo separada, usa todas as features
        self.data = dataframe.drop(columns=['timestamp']).values
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # Obtém a sequência de entrada e a previsão (pode adaptar conforme a tarefa)
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len]  # ou, se for previsão multi-step, ajuste aqui
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
