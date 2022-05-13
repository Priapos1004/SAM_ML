import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer

    bert_active = True
except:
    print(
        "build_embeddings(vec = 'bert') from data.bertembeddings cannot be used \n-> install 'sentence-transformers' to use this function"
    )
    bert_active = False

from tqdm.notebook import tqdm


class Embeddings_builder:
    def __init__(self, vec: str = "count", max_features: int = None):
        """
        @param:
            vec:
                'count' - CountVectorizer (default)
                'tfidf' - TfidfVectorizer
                'bert' - SentenceTransformer("quora-distilbert-multilingual")

            max_features - max_features of CountVectorizer or TfidfVectorizer
        """

        if bert_active and vec == "bert":
            print("using quora-distilbert-multilingual model as vectorizer")
            self.vectorizer = SentenceTransformer("quora-distilbert-multilingual")
            self.vec_type = vec

        elif vec == "count":
            print("using CountVectorizer as vectorizer")
            self.vectorizer = CountVectorizer(max_features=max_features)
            self.vec_type = vec

        elif vec == "tfidf":
            print("using TfidfVectorizer as vectorizer")
            self.vectorizer = TfidfVectorizer(max_features=max_features)
            self.vec_type = vec

        else:
            print(
                f"the entered vectorizer '{vec}' cannot be used --> using CountVectorizer as vectorizer"
            )
            self.vectorizer = CountVectorizer(max_features=max_features)
            self.vec_type = "count"

    def vectorize(self, data: pd.DataFrame, train_on: bool = True) -> pd.DataFrame:
        print("starting to create embeddings...")
        if self.vec_type == "bert":
            message_embeddings = [self.vectorizer.encode(str(i)) for i in tqdm(data)]
            emb_ar = np.asarray(message_embeddings)

        else:
            if train_on:
                emb_ar = self.vectorizer.fit_transform(data).toarray()
            else:
                emb_ar = self.vectorizer.transform(data).toarray()

        emb_df = pd.DataFrame(emb_ar)
        print("... embeddings created")

        return emb_df
