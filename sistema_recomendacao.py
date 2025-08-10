import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import requests
import io
import zipfile

print("="*60)
print("SISTEMA DE RECOMENDAÇÃO COM DECOMPOSIÇÃO MATRICIAL (SVD)")
print("="*60)

# ============================================================
# ETAPA 1: Carregamento de Dados com Múltiplas Estratégias
# ============================================================
print("\n[ETAPA 1] Obtendo dados do MovieLens...")

def load_dataset():
    # Verifica se já temos dados locais
    if os.path.exists('ml-latest-small/ratings.csv') and os.path.exists('ml-latest-small/movies.csv'):
        print("✅ Usando arquivos locais")
        return pd.read_csv('ml-latest-small/ratings.csv'), pd.read_csv('ml-latest-small/movies.csv')
    
    # Fontes de dados online
    sources = [
        {
            'name': 'GroupLens Oficial',
            'url': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
            'type': 'zip'
        },
        {
            'name': 'GitHub Mirror 1',
            'url': 'https://github.com/victor-santos10/recomendacao-movielens/raw/main/ml-latest-small.zip',
            'type': 'zip'
        },
        {
            'name': 'GitHub Mirror 2',
            'url': 'https://github.com/ageron/handson-ml2/raw/master/datasets/movielens/ml-latest-small.zip',
            'type': 'zip'
        },
        {
            'name': 'CSV Direto',
            'ratings': 'https://raw.githubusercontent.com/victor-santos10/recomendacao-movielens/main/ml-latest-small/ratings.csv',
            'movies': 'https://raw.githubusercontent.com/victor-santos10/recomendacao-movielens/main/ml-latest-small/movies.csv',
            'type': 'csv'
        }
    ]
    
    for source in sources:
        try:
            print(f"Tentando {source['name']}...")
            if source['type'] == 'zip':
                # Download do ZIP
                response = requests.get(source['url'], timeout=10)
                response.raise_for_status()
                
                # Salva temporariamente
                with open('temp_dataset.zip', 'wb') as f:
                    f.write(response.content)
                
                # Extrai
                with zipfile.ZipFile('temp_dataset.zip', 'r') as zip_ref:
                    zip_ref.extractall('.')
                
                # Remove temporário
                os.remove('temp_dataset.zip')
                print("✅ Download e extração completos")
                
            elif source['type'] == 'csv':
                # Download direto dos CSVs
                ratings = pd.read_csv(source['ratings'])
                movies = pd.read_csv(source['movies'])
                
                # Salva localmente
                os.makedirs('ml-latest-small', exist_ok=True)
                ratings.to_csv('ml-latest-small/ratings.csv', index=False)
                movies.to_csv('ml-latest-small/movies.csv', index=False)
            
            # Carrega dados
            ratings = pd.read_csv('ml-latest-small/ratings.csv')
            movies = pd.read_csv('ml-latest-small/movies.csv')
            print(f"✅ Sucesso com {source['name']}")
            return ratings, movies
            
        except Exception as e:
            print(f"⚠️ Falha com {source['name']}: {str(e)[:100]}...")
    
    # Se todas falharem, usa dados embutidos
    print("⚠️ Usando dataset de exemplo embutido")
    ratings_data = """userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
1,6,4.0,964982224
1,47,5.0,964983815
1,50,5.0,964982931
2,1,3.0,964982703
2,3,3.0,964982703
2,6,3.0,964982703
2,47,4.0,964983815
2,50,3.5,964982931
3,1,5.0,964982703
3,10,4.0,964981247
3,32,3.5,964982224
3,47,4.5,964983815
3,50,4.0,964982931"""
    
    movies_data = """movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
6,Heat (1995),Action|Crime|Thriller
10,GoldenEye (1995),Action|Adventure|Thriller
32,Twelve Monkeys (1995),Mystery|Sci-Fi|Thriller
47,Seven (a.k.a. Se7en) (1995),Mystery|Thriller
50,Usual Suspects, The (1995),Crime|Mystery|Thriller"""
    
    return pd.read_csv(io.StringIO(ratings_data)), pd.read_csv(io.StringIO(movies_data))

# Carrega os dados
try:
    ratings, movies = load_dataset()
    print(f"👉 {len(ratings)} avaliações de {len(ratings['userId'].unique())} usuários")
    print(f"👉 {len(movies)} filmes disponíveis")
except Exception as e:
    print(f"❌ Erro crítico ao carregar dados: {e}")
    exit()

# ============================================================
# ETAPA 2: Pré-processamento de Dados
# ============================================================
print("\n[ETAPA 2] Construindo matriz usuário-item...")

try:
    # Cria matriz de avaliações
    user_item_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)
    
    print(f"✅ Matriz criada: {user_item_matrix.shape[0]} usuários x {user_item_matrix.shape[1]} filmes")
    
    # Normalização por usuário
    user_ratings_mean = np.mean(user_item_matrix.values, axis=1)
    ratings_normalized = user_item_matrix.values - user_ratings_mean.reshape(-1, 1)
    
except Exception as e:
    print(f"❌ Erro no pré-processamento: {e}")
    exit()

# ============================================================
# ETAPA 3: Decomposição SVD (Álgebra Linear Aplicada)
# ============================================================
print("\n[ETAPA 3] Aplicando decomposição SVD...")

try:
    # Calcula número seguro de componentes
    k = min(50, min(user_item_matrix.shape)-1)
    if k < 5:
        k = min(user_item_matrix.shape)-1
    
    # Decomposição SVD
    U, sigma, Vt = svds(ratings_normalized, k=k)
    sigma = np.diag(sigma)
    
    # Reconstrução da matriz de predições
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(
        predicted_ratings,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )
    
    print(f"✅ SVD completo com k={k} componentes")
    
except Exception as e:
    print(f"❌ Erro no SVD: {e}")
    exit()

# ============================================================
# ETAPA 4: Avaliação do Modelo
# ============================================================
print("\n[ETAPA 4] Avaliando modelo...")

try:
    # Calcula RMSE apenas para avaliações existentes
    mask = user_item_matrix.values > 0
    rmse = np.sqrt(mean_squared_error(
        user_item_matrix.values[mask], 
        preds_df.values[mask]
    ))
    print(f"✅ RMSE do modelo: {rmse:.4f} (Quanto menor, melhor)")
    
except Exception as e:
    print(f"⚠️ Não foi possível calcular RMSE: {e}")

# ============================================================
# ETAPA 5: Visualização de Resultados
# ============================================================
print("\n[ETAPA 5] Gerando visualizações...")

try:
    # Gráfico de variância explicada
    explained_variance = np.cumsum(np.diag(sigma)) / np.sum(np.diag(sigma))
    plt.figure(figsize=(10, 6))
    plt.plot(explained_variance, 'o-')
    plt.title('Variância Explicada pelos Componentes do SVD')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Acumulada')
    plt.grid(True)
    plt.savefig('svd_variance.png')
    print("✅ Gráfico salvo como 'svd_variance.png'")
    
except Exception as e:
    print(f"⚠️ Não foi possível gerar gráfico: {e}")

# ============================================================
# ETAPA 6: Sistema de Recomendações Interativo
# ============================================================
print("\n" + "="*60)
print("SISTEMA PRONTO PARA RECOMENDAÇÕES!")
print("="*60)

def recommend_movies(user_id, n=10):
    try:
        # Filmes já avaliados pelo usuário
        user_rated = ratings[ratings['userId'] == user_id]['movieId']
        
        # Predições não avaliadas
        user_preds = preds_df.loc[user_id]
        recommendations = user_preds.drop(user_rated, errors='ignore')\
                                  .sort_values(ascending=False)\
                                  .head(n)
        
        # Junta com informações dos filmes
        return movies.merge(recommendations.to_frame(name='predicted_rating'), 
                          left_on='movieId', 
                          right_index=True)\
                   .sort_values('predicted_rating', ascending=False)
    
    except KeyError:
        # Usuário não encontrado - recomenda filmes populares
        print(f"⚠️ Usuário {user_id} não encontrado. Gerando recomendações populares")
        movie_ratings = ratings.groupby('movieId')['rating'].agg(['mean', 'count'])
        top_movies = movie_ratings[movie_ratings['count'] > 10].sort_values('mean', ascending=False).head(n)
        return movies.merge(top_movies, left_on='movieId', right_index=True)\
                     .sort_values('mean', ascending=False)\
                     .rename(columns={'mean': 'predicted_rating'})
    
    except Exception as e:
        print(f"⚠️ Erro ao gerar recomendações: {e}")
        return pd.DataFrame(columns=['movieId', 'title', 'genres', 'predicted_rating'])

# Interface do usuário
while True:
    try:
        user_id = input("\nDigite o ID do usuário (1-{}), 'pop' para filmes populares, ou 0 para sair: ".format(
            user_item_matrix.shape[0]))
        
        if user_id == '0':
            break
            
        if user_id == 'pop':
            # Mostra filmes populares
            movie_ratings = ratings.groupby('movieId')['rating'].agg(['mean', 'count'])
            top_movies = movie_ratings[movie_ratings['count'] > 10].sort_values('mean', ascending=False).head(10)
            result = movies.merge(top_movies, left_on='movieId', right_index=True)\
                           .sort_values('mean', ascending=False)
            
            print("\n🎬 TOP 10 FILMES POPULARES:")
            print(result[['title', 'genres', 'mean']].rename(columns={'mean': 'rating'}).to_string(index=False))
            continue
            
        user_id = int(user_id)
        n = int(input("Quantas recomendações deseja? (5-20): ") or 10)
        n = max(5, min(n, 20))
        
        recs = recommend_movies(user_id, n)
        
        if recs.empty:
            print("❌ Não foi possível gerar recomendações para este usuário")
        else:
            print(f"\n🎬 TOP {n} RECOMENDAÇÕES PARA USUÁRIO {user_id}:")
            print(recs[['title', 'genres', 'predicted_rating']].to_string(index=False))
            
            save = input("\nSalvar resultados em CSV? (s/n): ").lower()
            if save == 's':
                filename = f'recomendacoes_usuario_{user_id}.csv'
                recs.to_csv(filename, index=False)
                print(f"✅ Resultados salvos em '{filename}'")
            
    except ValueError:
        print("⚠️ Entrada inválida. Digite um número.")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")

print("\nSistema encerrado. Até logo!")