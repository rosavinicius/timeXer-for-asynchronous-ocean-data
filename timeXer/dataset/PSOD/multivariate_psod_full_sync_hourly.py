
import pandas as pd
import numpy as np
import os


# 'path': caminho do arquivo
# 'datetime_col': nome da coluna de timestamp no arquivo original
# 'value_cols_to_keep': lista das colunas de valor que queremos manter deste arquivo
# 'prefix': um prefixo para adicionar aos nomes das 'value_cols_to_keep' para torná-los únicos no DF final
input_files_info = [
    {
        'path': 'train/astronomical_tide.parquet',
        'datetime_col': 'datetime', # Supondo que seja 'datetime'
        'value_cols_to_keep': ['astronomical_tide'], # Supondo que a coluna de valor se chama 'tide'
        'prefix': 'astro_tide_'
    },
    {
        'path': 'train/current_praticagem.parquet',
        'datetime_col': 'datetime', # Supondo que seja 'time'
        'value_cols_to_keep': ['cross_shore_current'], # Supondo que a coluna de valor se chama 'current_speed'
        'prefix': 'curr_prat_'
    },
    {
        'path': 'train/sofs_praticagem.parquet',
        'datetime_col': 'datetime',
        'value_cols_to_keep': ['cross_shore_current', 'ssh'], # Múltiplas colunas
        'prefix': 'sofs_prat_'
    },
    {
        'path': 'train/waves_palmas.parquet',
        'datetime_col': 'datetime', # Supondo que seja 'date_time'
        'value_cols_to_keep': ['hs', 'tp', 'ws'], # Múltiplas colunas
        'prefix': 'waves_palm_'
    },
    {
        'path': 'train/wind_praticagem.parquet',
        'datetime_col': 'datetime',
        'value_cols_to_keep': ['vx', 'vy'], # Múltiplas colunas
        'prefix': 'wind_prat_'
    },
    { # Adicionando o ssh_praticagem original
        'path': 'train/ssh_praticagem.parquet',
        'datetime_col': 'datetime',
        'value_cols_to_keep': ['ssh'],
        'prefix': 'ssh_prat_' # Dando um prefixo para distingui-lo do 'ssh' do SOFS, se necessário
    }
]

output_csv_file = "multivariate_psod_full_sync_hourly.csv"
resample_freq = 'H' # Frequência horária para resampling
interpolation_method = 'linear' # Método de interpolação

processed_dfs = []

# --- Passo 1 e 2: Carregar, Pré-processar Colunas Múltiplas e Resamplear cada arquivo ---
print("--- Processando arquivos individuais ---")
for file_info in input_files_info:
    print(f"Processando: {file_info['path']}...")
    try:
        df = pd.read_parquet(file_info['path'])
        
        # Assegurar que a coluna de datetime exista e converter
        if file_info['datetime_col'] not in df.columns:
            print(f"  AVISO: Coluna de datetime '{file_info['datetime_col']}' não encontrada em {file_info['path']}. Pulando arquivo.")
            continue
        df[file_info['datetime_col']] = pd.to_datetime(df[file_info['datetime_col']])
        df = df.set_index(file_info['datetime_col'])
        df = df.sort_index()
        
        # Lidar com duplicatas no índice
        df = df[~df.index.duplicated(keep='first')]

        # Selecionar e Renomear colunas de valor
        cols_to_process = []
        rename_map = {}
        for original_col_name in file_info['value_cols_to_keep']:
            if original_col_name not in df.columns:
                print(f"  AVISO: Coluna de valor '{original_col_name}' não encontrada em {file_info['path']}. Pulando esta coluna.")
                continue
            new_col_name = f"{file_info['prefix']}{original_col_name}"
            rename_map[original_col_name] = new_col_name
            cols_to_process.append(original_col_name)
        
        if not cols_to_process:
            print(f"  AVISO: Nenhuma coluna de valor válida encontrada ou especificada para {file_info['path']}. Pulando arquivo.")
            continue
            
        df_subset = df[cols_to_process]
        df_renamed = df_subset.rename(columns=rename_map)
        
        # Resamplear o DataFrame que agora contém múltiplas colunas renomeadas
        df_resampled = df_renamed.resample(resample_freq).mean() # .mean() será aplicado a todas as colunas
        processed_dfs.append(df_resampled)
        print(f"  {file_info['path']} resampleado com colunas: {list(df_resampled.columns)} e adicionado.")
        
    except FileNotFoundError:
        print(f"  AVISO: Arquivo '{file_info['path']}' não encontrado. Pulando.")
    except Exception as e:
        print(f"  ERRO ao processar '{file_info['path']}': {e}. Pulando.")

if not processed_dfs:
    print("Nenhum arquivo foi processado com sucesso. Encerrando.")
    exit()

# --- Passo 3: Mesclar todos os DataFrames resampleados ---
print(f"\n--- Mesclando {len(processed_dfs)} DataFrames resampleados ---")
df_merged = pd.concat(processed_dfs, axis=1)
print(f"DataFrame mesclado criado com {df_merged.shape[0]} linhas e {df_merged.shape[1]} colunas de valor.")
print("Primeiras linhas do DataFrame mesclado (antes da interpolação):")
print(df_merged.head())

# --- Passo 4: Interpolar Valores Ausentes ---
print(f"\n--- Interpolando valores ausentes usando método '{interpolation_method}' ---")
df_synced = df_merged.interpolate(method=interpolation_method)
df_synced = df_synced.ffill().bfill() # Preencher NaNs restantes no início ou fim
print("Interpolação concluída.")
print("Verificando valores ausentes após interpolação final (sumário):")
print(df_synced.isnull().sum().to_string()) # .to_string() para mostrar tudo se houver muitas colunas

# --- Passo 5: Adicionar Features de Tempo Cíclicas ---
print("\n--- Adicionando Features de Tempo Cíclicas ---")
df_synced_with_time = df_synced.reset_index() 
# Renomear a coluna de timestamp (anteriormente o índice) para 'date'
# O nome do índice pode variar, então tentamos capturar 'index' ou o nome explícito
idx_name = df_synced_with_time.columns[0] if df_synced_with_time.columns[0] != df_synced.columns[0] else 'index' # Heurística
df_synced_with_time.rename(columns={idx_name: 'date'}, inplace=True)

df_synced_with_time['hour'] = df_synced_with_time['date'].dt.hour
df_synced_with_time['dayofweek'] = df_synced_with_time['date'].dt.dayofweek

df_synced_with_time['hour_sin'] = np.sin(2 * np.pi * df_synced_with_time['hour'] / 24.0)
df_synced_with_time['hour_cos'] = np.cos(2 * np.pi * df_synced_with_time['hour'] / 24.0)
df_synced_with_time['dayofweek_sin'] = np.sin(2 * np.pi * df_synced_with_time['dayofweek'] / 7.0)
df_synced_with_time['dayofweek_cos'] = np.cos(2 * np.pi * df_synced_with_time['dayofweek'] / 7.0)

df_final = df_synced_with_time.drop(columns=['hour', 'dayofweek'])
print("Features de tempo cíclicas adicionadas.")

# --- Passo 6: Preparar DataFrame Final e Salvar em CSV ---
# Garantir que 'date' seja a primeira coluna
all_value_cols = list(df_synced.columns) # Colunas de valor originais (já prefixadas)
time_feature_cols = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
cols_final_order = ['date'] + all_value_cols + time_feature_cols
df_final = df_final[cols_final_order]


print("\n--- DataFrame Final Multivariado Síncrono (primeiras linhas) ---")
print(df_final.head())
print("\n--- Informações do DataFrame Final ---")
df_final.info()
print("\n--- Nomes das Colunas Finais ---")
print(list(df_final.columns))


print(f"\n--- Salvando DataFrame final para: {output_csv_file} ---")
try:
    df_final.to_csv(output_csv_file, index=False)
    print("Arquivo CSV multivariado salvo com sucesso!")
    num_features_for_enc_in = len(df_final.columns) -1 # -1 por causa da coluna 'date'
    print(f"\nEste arquivo pode ser usado com o TimeXer.")
    print(f"Lembre-se de ajustar o parâmetro '--enc_in' (e dec_in, c_out) para: {num_features_for_enc_in}")
    print(f"Especifique o '--target' com o nome da coluna prefixada (ex: 'sofs_prat_ssh' ou 'ssh_prat_ssh').")

except Exception as e:
    print(f"ERRO ao salvar o arquivo CSV: {e}")