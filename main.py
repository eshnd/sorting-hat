from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def search(inn):
    titles = ["", "", ""]
    descps = ["", "", ""]

    print("converting data to vectors")
    vec = TfidfVectorizer()
    vector_descps = vec.fit_transform(descps)
  
    print("fitting vectors to the titles")
    main_m = KNeighborsClassifier(n_neighbors=1)
    main_m.fit(vector_descps, titles)
    

    def search_model(ina):
        print("converting input to vectors")
        ina_vec = vec.transform([ina])
        print("predicting the best fitting title")
        prediction = main_m.predict(ina_vec)
        return prediction[0]

    final = search_model(inn)
    return final

inp = input("search: ")
print(search(inp))
