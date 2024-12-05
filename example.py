from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def search(inn):
    titles = ["phineas and ferb", "gravity falls", "adventure time"]
    descps = ["2 two boys phineas and ferb building things", "2 two kids go to grunkle gravity falls solve mystery mysteries", "finn the human and jake the dog"]

    print("converting data to vectors")
    vec = TfidfVectorizer()
    vector_descps = vec.fit_transform(descps)
  
    print("fitting vectors to the titles")
    main_m = KNeighborsClassifier(n_neighbors=1)
    main_m.fit(vector_descps, titles)
    

    def search_model(ina):
        print("converting input to vectors")
        ina_vec = vec.transform([ina])
        print("predicting the best fitting title (this takes the longest due to the complex calculations. for each word, it uses a modified distance formula and it compares it to the vectors/tokens in the training data.)")
        prediction = main_m.predict(ina_vec)
        return prediction[0]

    final = search_model(inn)
    return final

inp = input("search (describe with as many details as possible): ")
print(search(inp))
