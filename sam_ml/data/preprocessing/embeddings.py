import concurrent.futures

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm.auto import tqdm

from sam_ml.config import setup_logger

from .main_data import DATA

logger = setup_logger(__name__)


class Embeddings_builder(DATA):
    """ Vectorizer Wrapper class """

    def __init__(self, algorithm: str = "tfidf", **kwargs):
        """
        @param:
            algorithm:
                'count': CountVectorizer (default)
                'tfidf': TfidfVectorizer
                'bert': SentenceTransformer("quora-distilbert-multilingual")

            **kwargs:
                additional parameters for CountVectorizer or TfidfVectorizer
        """
        if algorithm == "bert":
            vectorizer = SentenceTransformer("quora-distilbert-multilingual")
        elif algorithm == "count":
            vectorizer = CountVectorizer(**kwargs)
        elif algorithm == "tfidf":
            vectorizer = TfidfVectorizer(**kwargs)
        else:
            raise ValueError(f"algorithm='{algorithm}' is not supported")
        
        super().__init__(algorithm, vectorizer)

    @staticmethod
    def params() -> dict:
        """
        @return:
            possible values for the parameters
        """
        param = {"vec": ["bert", "count", "tfidf"]}
        return param

    def get_params(self, deep: bool = True):
        class_params = {"vec": self.algorithm}
        if self.algorithm != "bert":
            return class_params | self.transformer.get_params(deep)
        return class_params

    def set_params(self, **params):
        if self.algorithm == "bert":
            self.transformer = SentenceTransformer("quora-distilbert-multilingual", **params)
        else:
            self.transformer.set_params(**params)
        return self
    
    def create_parallel_bert_embeddings(self, content: list) -> list:
        logger.debug("Going to parallel process embedding creation")

        # Create a progress bar
        pbar = tqdm(total=len(content), desc="Bert Embeddings")

        # Define a new function that updates the progress bar after each embedding
        def get_embedding_and_update(text: str) -> list:
            pbar.update()
            return self.transformer.encode(text)
        
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            content_embeddings = list(executor.map(get_embedding_and_update, content))

        # Close the progress bar
        pbar.close()

        return content_embeddings

    def vectorize(self, data: pd.Series, train_on: bool = True) -> pd.DataFrame:
        """
        @params:
            data: pandas Series
            train_on: shall the vectorizer fit before transform
        @return:
            pandas Dataframe with vectorized data
        """
        indices = data.index
        logger.debug("creating embeddings - started")
        if self.algorithm == "bert":
            message_embeddings = self.create_parallel_bert_embeddings(list(data))
            emb_ar = np.asarray(message_embeddings)

        else:
            if train_on:
                emb_ar = self.transformer.fit_transform(data).toarray()
            else:
                emb_ar = self.transformer.transform(data).toarray()

        emb_df = pd.DataFrame(emb_ar, index=indices).add_suffix("_"+data.name)
        logger.debug("creating embeddings - finished")

        return emb_df
