import pandas as pd
import numpy as np
import os

# --- Configurações ---
input_parquet_file = "train/ssh_praticagem.parquet"
output_csv_file = "ssh_psod_sync_hourly.csv" # Nome do arquivo CSV para o TimeXer
target_column = 'ssh' # Sua coluna alvo
datetime_column = 'datetime' # Nome da sua coluna de data/hora

# --- Passo 0: Carregar os Dados ---
print(f"Carregando dados de: {input_parquet_file}")
try:
    df_original = pd.read_parquet(input_parquet_file)
    print("Dados carregados com sucesso.")
except FileNotFoundError:
    print(f"ERRO: Arquivo '{input_parquet_file}' não encontrado. Verifique o caminho.")
    exit()
except Exception as e:
    print(f"ERRO ao carregar o arquivo Parquet: {e}")
    exit()

print("\n--- DataFrame Original (primeiras linhas) ---")
print(df_original.head())
print("\n--- Informações do DataFrame Original ---")
df_original.info()

# --- Passo 1: Garantir Tipo 'datetime' e Definir como Índice ---
print(f"\n--- Processando coluna '{datetime_column}' ---")
if datetime_column not in df_original.columns:
    print(f"ERRO: Coluna '{datetime_column}' não encontrada no DataFrame.")
    exit()

df_original[datetime_column] = pd.to_datetime(df_original[datetime_column])
df_original = df_original.set_index(datetime_column)
# Ordenar pelo índice de tempo é uma boa prática antes do resampling
df_original = df_original.sort_index()
print(f"Coluna '{datetime_column}' definida como índice e ordenada.")

# Verificar se a coluna alvo existe
if target_column not in df_original.columns:
    print(f"ERRO: Coluna alvo '{target_column}' não encontrada.")
    exit()

# Selecionar apenas a coluna alvo para resampling inicialmente, se houver outras
# Se você tiver outras colunas de features que quer resamplear junto, inclua-as aqui.
# Por agora, focaremos na 'ssh' conforme o exemplo.
df_to_resample = df_original[[target_column]]

# --- Passo 2: Resample para Frequência Horária e Interpolar ---
print("\n--- Resampling para frequência horária ('H') e Interpolação ---")
# Se múltiplos pontos caírem na mesma hora, .mean() tira a média.
# Se não houver pontos, será NaN inicialmente.
df_resampled = df_to_resample.resample('H').mean()

# Interpolar valores NaN usando interpolação linear
# Outros métodos: 'ffill' (forward fill), 'bfill' (backward fill), 'spline', 'polynomial' (com order=...)
df_sync = df_resampled.interpolate(method='linear')

# Preencher NaNs restantes no início ou fim (se a interpolação linear não cobrir)
# ffill preenche com o último valor válido; bfill com o próximo.
df_sync = df_sync.ffill().bfill()

print("Resampling e interpolação concluídos.")
print(f"Número de linhas após resampling: {len(df_sync)}")
print("Verificando valores ausentes após interpolação e preenchimento:")
print(df_sync.isnull().sum())

# Voltar 'datetime' (agora 'date') para ser uma coluna, como TimeXer espera
df_sync = df_sync.reset_index()
# Renomear a coluna de timestamp para 'date' como o Dataset_Custom do TimeXer parece esperar
df_sync.rename(columns={datetime_column: 'date'}, inplace=True)


print("\n--- DataFrame Sincronizado (primeiras linhas) ---")
print(df_sync.head())

# --- Passo 3: Adicionar Features de Tempo Cíclicas ---
print("\n--- Adicionando Features de Tempo Cíclicas ---")
df_sync['hour'] = df_sync['date'].dt.hour
df_sync['dayofweek'] = df_sync['date'].dt.dayofweek # Segunda=0, Domingo=6

# Codificações seno/cosseno
df_sync['hour_sin'] = np.sin(2 * np.pi * df_sync['hour'] / 24.0)
df_sync['hour_cos'] = np.cos(2 * np.pi * df_sync['hour'] / 24.0)
df_sync['dayofweek_sin'] = np.sin(2 * np.pi * df_sync['dayofweek'] / 7.0)
df_sync['dayofweek_cos'] = np.cos(2 * np.pi * df_sync['dayofweek'] / 7.0)

# Opcional: Remover colunas intermediárias 'hour' e 'dayofweek' se não forem usadas diretamente
df_final = df_sync.drop(columns=['hour', 'dayofweek'])
print("Features de tempo cíclicas adicionadas.")

# --- Passo 4: Preparar DataFrame Final e Ordem das Colunas ---
# O Dataset_Custom do TimeXer espera 'date' como primeira coluna e depois as features.
# Ele internamente move a coluna 'target' para o final (se não for univariado 'S').
# Vamos organizar as colunas para clareza: 'date', features de tempo, coluna alvo.
feature_cols = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
# Garante que a coluna target_column ('ssh') venha depois das features de tempo
df_final = df_final[['date'] + feature_cols + [target_column]]

print("\n--- DataFrame Final com Features (primeiras linhas) ---")
print(df_final.head())
print("\n--- Informações do DataFrame Final ---")
df_final.info()

# --- Passo 5: Salvar para CSV ---
print(f"\n--- Salvando DataFrame final para: {output_csv_file} ---")
try:
    df_final.to_csv(output_csv_file, index=False)
    print("Arquivo CSV salvo com sucesso!")
    print(f"Este arquivo pode ser usado com o TimeXer, definindo:")
    print(f"  --data custom")
    print(f"  --root_path ./ (ou o diretório onde você salvou o CSV)")
    print(f"  --data_path {output_csv_file}")
    print(f"  --features M")
    print(f"  --target {target_column}")
    num_features_for_enc_in = len(df_final.columns) -1 # -1 por causa da coluna 'date'
    print(f"  --enc_in {num_features_for_enc_in} (e provavelmente --dec_in e --c_out com o mesmo valor para M)")
except Exception as e:
    print(f"ERRO ao salvar o arquivo CSV: {e}")