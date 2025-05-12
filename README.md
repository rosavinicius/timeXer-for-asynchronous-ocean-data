# Adaptação do Modelo TimeXer para Previsão em Séries Temporais Oceanográficas Assíncronas (Dataset PSOD)

## Visão Geral 
Este repositório documenta o processo de adaptação e utilização do modelo de previsão de séries temporais TimeXer com o dataset PSOD (Port Santos Oceanographic Dataset). O PSOD é um conjunto de dados oceanográficos multivariados caracterizado por sua natureza assíncrona, onde diferentes sensores e variáveis são amostrados em frequências e timestamps irregulares.
o dataset se encontra no repositório infracitado: https://github.com/C4AI/psod_dataset

O objetivo principal deste projeto foi explorar metodologias para permitir que um modelo avançado como o TimeXer, que tipicamente espera dados síncronos, pudesse ser treinado e utilizado para realizar previsões com os dados desafiadores do PSOD.

## Desafios encontrados
O desafio fundamental reside na incompatibilidade entre a natureza dos dados e a expectativa do modelo:

Dataset PSOD:
- Assíncrono: Diferentes variáveis (ex: altura significativa da onda, velocidade do vento, nível do mar, maré astronômica) são coletadas em momentos distintos e com intervalos irregulares.

Modelo TimeXer (e Modelos Similares):
- Expectativa de Sincronicidade: A maioria dos modelos de forecasting de séries temporais baseados em Transformers ou RNNs, incluindo o TimeXer, são projetados para operar sobre dados síncronos. Isso significa que eles          esperam um valor para cada variável em cada passo de tempo discreto e regular na sequência de entrada.
- Estrutura Sequencial Regular: Mecanismos como positional encodings e a própria estrutura de processamento sequencial em janelas fixas dependem dessa regularidade.
  Alimentar diretamente dados assíncronos em um modelo síncrono não é viável sem uma etapa de adaptação (seja dos dados ou do modelo).

## Metodologia Adotada: Sincronização e Engenharia de Features
A principal estratégia adotada neste projeto foi adaptar os dados do PSOD para o formato síncrono esperado pelo TimeXer. Esta abordagem, embora possa introduzir aproximações, é frequentemente o caminho mais prático para utilizar modelos "state-of-the-art" existentes sem requerer modificações profundas em sua arquitetura interna.

O processo de preparação dos dados envolveu as seguintes etapas:

1. **Carregamento Individual dos Datasets:**

- Os dados brutos do PSOD são provenientes de múltiplos arquivos .parquet, cada um podendo conter uma ou mais variáveis e sua respectiva coluna de timestamp.
- Arquivos considerados: astronomical_tide.parquet, current_praticagem.parquet, sofs_praticagem.parquet (com colunas 'datetime', 'cross_shore_current', 'ssh'), waves_palmas.parquet (com 'datetime', 'hs', 'tp', 'ws'), wind_praticagem.parquet (com 'datetime', 'vx', 'vy') e ssh_praticagem.parquet.

2. **Processamento Individual de Cada Arquivo:**

- Conversão da coluna de timestamp para o formato datetime do pandas.
- Definição do timestamp como índice do DataFrame.
- Ordenação cronológica dos dados.
- Remoção de timestamps duplicados (mantendo o primeiro).
- Seleção das colunas de valor relevantes de cada arquivo.
- Renomeação de Colunas com Prefixo: Para garantir a unicidade dos nomes das colunas após a fusão de múltiplos arquivos (especialmente aqueles com múltiplas variáveis ou nomes de colunas comuns como 'ssh'), um prefixo baseado no nome/fonte do arquivo foi adicionado a cada coluna de valor (ex: sofs_prat_ssh, waves_palm_hs).

3. **Resampling para Frequência Comum: h**

- Todos os DataFrames processados individualmente foram resampleados para uma frequência horária comum ('H').
- A função de agregação .mean() foi utilizada durante o resampling para consolidar pontos de dados que caíssem dentro do mesmo intervalo horário. Horas sem dados originais resultaram em NaN.

4. **Fusão dos Datasets Resampleados:**

- Os DataFrames resampleados (cada um agora com um índice de tempo horário regular e suas respectivas colunas de valor prefixadas) foram concatenados (pd.concat(..., axis=1)) em um único DataFrame multivariado. Este DataFrame continha todas as variáveis de interesse alinhadas na nova grade de tempo horária, com NaNs onde as séries originais não coincidiam.

5. **Interpolação de Valores Ausentes:**

- Para preencher os NaNs no DataFrame multivariado e síncrono, foi aplicada a interpolação linear (.interpolate(method='linear')).
- Valores NaN remanescentes no início ou fim do dataset (não cobertos pela interpolação linear) foram preenchidos usando ffill() (forward fill) seguido por bfill() (backward fill).

6. **Engenharia de Features de Tempo Cíclicas:**

- Com base na coluna de timestamp síncrona (agora chamada 'date'), foram criadas features adicionais para ajudar o modelo a capturar padrões temporais cíclicos:
hour_sin, hour_cos: Representação seno/cosseno da hora do dia.
- dayofweek_sin, dayofweek_cos: Representação seno/cosseno do dia da semana.
- Essas transformações ajudam o modelo a entender a proximidade entre, por exemplo, 23:00 e 00:00.

7. **Geração do Arquivo CSV Final:**

- O DataFrame final, contendo a coluna 'date', todas as variáveis de processo resampleadas, interpoladas e prefixadas, e as features de tempo cíclicas, foi salvo como um único arquivo CSV (multivariate_psod_full_sync_hourly.csv ou ssh_psod_sync_hourly.csv para o caso univariado com features de tempo). Este arquivo serve como entrada para o Dataset_Custom do TimeXer.

## Alternativas Consideradas (e Porque Não Adotadas Neste Projeto para TimeXer)
**Tratamento Direto de Dados Assíncronos com Time Embeddings (ex: Time2Vec)**

Uma abordagem alternativa e teoricamente mais "pura" seria modificar o modelo para que ele pudesse processar os dados assíncronos brutos, sem a etapa de sincronização. Técnicas como Time2Vec, que aprendem representações vetoriais para timestamps, são promissoras para este fim.

**Motivação para Não Seguir Este Caminho (com o TimeXer existente):**

- Complexidade de Modificação do Modelo: O TimeXer, como outros modelos baseados em Transformers, possui uma arquitetura que assume sequências de entrada de tamanho fixo e utiliza positional encodings baseados na posição discreta dentro dessas sequências. A integração efetiva do Time2Vec para lidar com timestamps irregulares e eventos esparsos exigiria:
  1. Reformulação do Carregamento de Dados e Batching: Para lidar com sequências de eventos de comprimento variável ou janelas de tempo com número variável de eventos, necessitando de padding e mascaramento complexos.
  2. Substituição/Adaptação dos Positional Encodings: Os encodings posicionais padrão teriam que ser substituídos ou complementados significativamente pelos time embeddings.
  3. Atenção Sensível ao Tempo (Time-Aware Attention): Idealmente, os mecanismos de atenção do Transformer precisariam ser modificados para levar em conta explicitamente os intervalos de tempo variáveis entre os eventos ao calcular os scores de atenção.
  4. Escopo do Projeto: O objetivo inicial era fazer o TimeXer funcionar com os dados do PSOD. Modificar a arquitetura central de um modelo complexo como o TimeXer impede de usá-lo "as it is" (ou com as configurações padrão para datasets customizados), aproveitando sua arquitetura já validada.
  5. Embora o uso de time embeddings em dados assíncronos seja uma direção valiosa, sua implementação efetiva geralmente requer que o modelo seja co-projetado para essa finalidade.

Modelo Utilizado
TimeXer: Modelo de previsão de séries temporais.
Repositório Original: https://github.com/thuml/TimeXer

# Execução dos experimentos 

**pré-requisitos** 
- **ambiente python**
   Configure um ambiente python na versão 3.8. Por conveniência, execute: 
    
```
pip install -r requirements.txt
```
1. **Preparação dos Dados**

Utilize o script Python fornecido neste repositório (``ssh_psod_sync_hourly.py`` - para uma única varivável - ou ``multivariate_psod_full_sync_hourly`` - para todas as variáveis) para processar o arquivos .parquet brutos.

Este script realizará o carregamento, tratamento de múltiplas colunas, prefixação, resampling horário, concatenação, interpolação e adição de features de tempo cíclicas.

**Entrada**: Os arquivos .parquet brutos.

**Saída**: Um arquivo CSV síncrono no diretório: ./dataset/PSOD/

```
python ssh_psod_sync_hourly.py
```
ou 

```
multivariate_psod_full_sync_hourly.py
```

2. **Treinamento do Modelo TimeXer**

- Utilize o script shell (.sh) fornecido neste repositório (PSOD_multivariate.sh ou PSOD_ssh.sh) para executar o treinamento.
- Para usuários de Mac M1/M2/M3: Certifique-se de que linhas como export CUDA_VISIBLE_DEVICES estejam comentadas ou removidas. O parâmetro --use_gpu 1 deve ser suficiente para o PyTorch tentar usar o backend MPS. O flag --gpu X pode precisar ser comentado.
  
- Execução:
```
# Navegue até o diretório raiz do repositório TimeXer
bash ./scripts/long_term_forecast/PSOD_script/PSOD_multivariate.sh
```

## Resultados 
- Para o modelo executado através do script ``PSOD_ssh.sh`` (com o dataset com uma única varivável de interesse - ssh)
obteve-se as seguintes métricas no conjunto de teste:

  - MSE: 0.0454
  - MAE: 0.1344

- Para o modelo executado através do script ``PSOD_multivariate.sh`` obteve-se as seguintes métricas no conjunto de teste:

  - MSE: 0.3165
  - MAE: 0.3455

* O erro AttributeError: module 'torch.backends.mps' has no attribute 'empty_cache' ocorreu no final da execução em ambiente Mac M1. Isso acontece porque o script run.py tenta chamar uma função de limpeza de cache específica da CUDA que não existe para o backend MPS. Este erro ocorre após o cálculo das métricas e pode ser corrigido modificando a respectiva linha no run.py para não chamar empty_cache quando estiver usando MPS.

