# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 15:04:26 2021

@author: Deven Shetty
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data = pd.read_csv("movie_dataset.csv")
result = data[["keywords","cast", "genres", "director" ]]
values = {"keywords":" ","cast": " ", "genres": " ", "director": " "}
result = result.fillna(value = values)
result.insert(4,"combine_feature", result["keywords"]+" " + result["cast"]+ " "  + result["genres"]+" "  + result["director"] , True)

count = CountVectorizer()
count = count.fit_transform(result["combine_feature"])
count = count.toarray()
similarity = cosine_similarity(count)

user = "Avatar"
def ind_of_movie(user):
    indexed = data[data["title"]==user].index.values
    print(indexed)
    return indexed
indexed = ind_of_movie(user)

sorts = np.sort(similarity[0])[::-1]

final_value = sorts[1]
ans = np.where(similarity[0]==final_value)
final_index= ans[0]
print(data["title"][final_index])
