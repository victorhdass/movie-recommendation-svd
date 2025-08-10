# 🎬 Sistema de Recomendação com Decomposição SVD

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema de recomendação de filmes utilizando decomposição matricial (SVD) implementado em Python.

![Exemplo de Recomendações](docs/recommendations_example.png)

## 🚀 Recursos
- Decomposição SVD para sistemas de recomendação
- Interface interativa para geração de recomendações
- Carregamento robusto de dados com múltiplas fontes
- Visualização da variância explicada pelos componentes
- Avaliação de desempenho com métrica RMSE

## ⚙️ Instalação
```bash
git clone https://github.com/seu-usuario/movie-recommendation-svd.git
cd movie-recommendation-svd

python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## 🖥️ Execução
```bash
cd src
python recommendation_system.py
```

## 🧠 Teoria Matemática
A decomposição SVD (Singular Value Decomposition) é definida como:

$$ A = U \Sigma V^T $$

Onde:
- $A$ é a matriz original de avaliações
- $U$ contém os autovetores da matriz $AA^T$
- $\Sigma$ é uma matriz diagonal com valores singulares
- $V^T$ contém os autovetores da matriz $A^TA$

## 📊 Resultados
| Métrica | Valor |
|---------|-------|
| RMSE    | 0.845 |
| Componentes SVD | 50 |

![Variância Explicada](docs/svd_variance.png)

## 🤝 Contribuição
Contribuições são bem-vindas! Siga estes passos:
1. Faça um fork do projeto
2. Crie sua branch (`git checkout -b feature/nova-feature`)
3. Faça commit das alterações (`git commit -m 'Adiciona nova feature'`)
4. Faça push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request
