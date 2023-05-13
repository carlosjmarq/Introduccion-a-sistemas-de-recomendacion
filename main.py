from cmath import nan
import re
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Tuple, Any, Callable, cast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

pelis_movielens = pd.read_csv("MovieLens_con_argumento.csv")

# print(pelis_movielens.head())

# * traductor de movieId a numero de fila

dicc_indice_movie = pelis_movielens["movieId"].to_dict()
dicc_movieid_indice = {valor: clave for clave, valor in dicc_indice_movie.items()}

# print(dicc_movieid_indice)

# * inicio del procesamiento de daots de los argumentos

def limpiar_argumentos(argumento: str) -> str:
	arg = argumento.lower()
	return re.sub(r"\d+", "", arg)

contador_argumento = CountVectorizer(preprocessor=limpiar_argumentos, min_df=5)

arguments_bag_of_words = (contador_argumento.fit_transform(pelis_movielens["argumento"].to_numpy())).toarray()

sorter: Callable[[Tuple[Any, int]], int] = lambda x: x[1]
columnas_argumentos = [t for t, i in sorted(contador_argumento.vocabulary_.items(), key=sorter)]

generos_bag_of_words_df = pd.DataFrame(arguments_bag_of_words, columns=columnas_argumentos, index=pelis_movielens['title'])
print(generos_bag_of_words_df.head())

def tokenizador_generos(string_generos):
	generos_separados = string_generos.split('|')
	resultado: List[str] = []
	for size in [1,2]:
		combs = ["Generos -" + "|".join(sorted(tupla))
			for tupla in combinations(generos_separados, r=size)]
		resultado = resultado + combs
	return sorted(resultado)

contador_generos = CountVectorizer(tokenizer= tokenizador_generos, token_pattern=None, lowercase=False)

contador_generos.fit(pelis_movielens["genres"])

generos_bag_of_words = (contador_generos.fit_transform(pelis_movielens["genres"].to_numpy())).toarray()

columnas_generos = [t for t, i in sorted(contador_generos.vocabulary_.items(), key=sorter)]

generos_bag_of_words_df = pd.DataFrame(generos_bag_of_words, columns=columnas_generos, index=pelis_movielens['title'])
print(generos_bag_of_words_df.head())

#  se concatenan ambos dataframes

total_bag_of_words = np.hstack((arguments_bag_of_words, generos_bag_of_words))

total_bag_of_words_df = pd.DataFrame(total_bag_of_words, columns=columnas_argumentos+columnas_generos, index=pelis_movielens['title'])
print(total_bag_of_words_df.head())


# Aplicacion de TF-IDF

tf_idf = TfidfTransformer()

tf_idf_pelis = tf_idf.fit_transform(total_bag_of_words_df).toarray()
tf_idf_pelis_df = pd.DataFrame(tf_idf_pelis, columns=columnas_argumentos+columnas_generos, index=pelis_movielens['title'])
print(tf_idf_pelis_df.head())

# aplicacion de cosine similarity

cosine_sim = cosine_similarity(tf_idf_pelis)
matriz_similaridades_df =  pd.DataFrame(cosine_sim, columns=pelis_movielens['title'], index=pelis_movielens['title'])
# se hace nula la diagonal
np.fill_diagonal(matriz_similaridades_df.values, nan)
print(matriz_similaridades_df.head())

orden_pelis_por_fila = np.argsort((-cosine_sim), axis=1)

cosine_sims_ordenadas = np.sort(-cosine_sim, axis=1)

# funcion para buscar en la similaridades
def top_k_pelis(movieId, k = 5):
	indice = dicc_movieid_indice[movieId]
	fila_peliculas = orden_pelis_por_fila[indice]
	fila_similaridades = cosine_sims_ordenadas[indice]

	top_k = fila_peliculas[:k]
	similaridades = fila_similaridades[:k]

	top_k_df = pelis_movielens.loc[top_k].copy()
	top_k_df["similaridad"] = similaridades

	return top_k_df

print(top_k_pelis(6283, 15))