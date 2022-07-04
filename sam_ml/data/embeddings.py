import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer

    bert_active = True
except:
    bert_active = False

from tqdm.auto import tqdm


class Embeddings_builder:
    def __init__(self, vec: str = "count", console_out: bool = True, **kwargs):
        """
        @param:
            vec:
                'count': CountVectorizer (default)
                'tfidf': TfidfVectorizer
                'bert': SentenceTransformer("quora-distilbert-multilingual")

            You can change all parameters from CountVectorizer and TfidfVectorizer
        """
        self.console_out = console_out
        if bert_active and vec == "bert":
            if self.console_out:
                print("using quora-distilbert-multilingual model as vectorizer")
            self.vectorizer = SentenceTransformer("quora-distilbert-multilingual")
            self.vec_type = vec

        elif not bert_active and vec == "bert":
            print("build_embeddings(vec = 'bert') from data.bertembeddings cannot be used \n-> install 'sentence-transformers' to use this function")

        elif vec == "count":
            if self.console_out:
                print("using CountVectorizer as vectorizer")
            self.vectorizer = CountVectorizer(**kwargs)
            self.vec_type = vec

        elif vec == "tfidf":
            if self.console_out:
                print("using TfidfVectorizer as vectorizer")
            self.vectorizer = TfidfVectorizer(**kwargs)
            self.vec_type = vec

        else:
            if self.console_out:
                print(f"the entered vectorizer '{vec}' cannot be used --> using CountVectorizer as vectorizer")
            self.vectorizer = CountVectorizer()
            self.vec_type = "count"

    def vectorize(self, data: pd.Series, train_on: bool = True) -> pd.DataFrame:
        indices = data.index
        if self.console_out:
            print("starting to create embeddings...")
        if self.vec_type == "bert":
            message_embeddings = [self.vectorizer.encode(str(i)) for i in tqdm(data, desc="Bert Embeddings")]
            emb_ar = np.asarray(message_embeddings)

        else:
            if train_on:
                emb_ar = self.vectorizer.fit_transform(data).toarray()
            else:
                emb_ar = self.vectorizer.transform(data).toarray()

        emb_df = pd.DataFrame(emb_ar, index=indices).add_suffix("_"+data.name)
        if self.console_out:
            print("... embeddings created")

        return emb_df
