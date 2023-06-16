import concurrent.futures

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm.auto import tqdm

from sam_ml.config import setup_logger

logger = setup_logger(__name__)


class Embeddings_builder:
    """ Vectorizer Wrapper class """

    def __init__(self, vec: str = "count", **kwargs):
        """
        @param:
            vec:
                'count': CountVectorizer (default)
                'tfidf': TfidfVectorizer
                'bert': SentenceTransformer("quora-distilbert-multilingual")

            **kwargs:
                additional parameters for CountVectorizer or TfidfVectorizer
        """
        self.vec_type = vec
        self._grid: dict[str, list] = {} # for pipeline structure

        if vec == "bert":
            logger.debug("using quora-distilbert-multilingual model as vectorizer")
            self.vectorizer = SentenceTransformer("quora-distilbert-multilingual")

        elif vec == "count":
            logger.debug("using CountVectorizer as vectorizer")
            self.vectorizer = CountVectorizer(**kwargs)

        elif vec == "tfidf":
            logger.debug("using TfidfVectorizer as vectorizer")
            self.vectorizer = TfidfVectorizer(**kwargs)

        else:
            logger.error(f"the entered vectorizer '{vec}' cannot be used --> using CountVectorizer as vectorizer")
            self.vectorizer = CountVectorizer()
            self.vec_type = "count"

    def __repr__(self) -> str:
        vec_params: str = ""
        param_dict = self.get_params(False)
        for key in param_dict:
            if type(param_dict[key]) == str:
                vec_params += key+"='"+str(param_dict[key])+"', "
            else:
                vec_params += key+"="+str(param_dict[key])+", "
        return f"Embeddings_builder({vec_params})"

    @staticmethod
    def params() -> dict:
        """
        @return:
            possible values for the parameters
        """
        param = {"vec": ["bert", "count", "tfidf"]}
        return param

    def get_params(self, deep: bool = True):
        class_params = {"vec": self.vec_type}
        if self.vec_type != "bert":
            return class_params | self.vectorizer.get_params(deep)
        return class_params | {"model_name_or_path": "quora-distilbert-multilingual"}

    def set_params(self, **params):
        if self.vec_type == "bert":
            self.vectorizer = SentenceTransformer("quora-distilbert-multilingual", **params)
        else:
            self.vectorizer.set_params(**params)
        return self
    
    def create_parallel_bert_embeddings(self, content: list) -> list:
        logger.debug("Going to parallel process embedding creation")

        # Create a progress bar
        pbar = tqdm(total=len(content), desc="Bert Embeddings")

        # Define a new function that updates the progress bar after each embedding
        def get_embedding_and_update(text: str) -> list:
            pbar.update()
            return self.vectorizer.encode(text)
        
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
        if self.vec_type == "bert":
            message_embeddings = self.create_parallel_bert_embeddings(list(data))
            emb_ar = np.asarray(message_embeddings)

        else:
            if train_on:
                emb_ar = self.vectorizer.fit_transform(data).toarray()
            else:
                emb_ar = self.vectorizer.transform(data).toarray()

        emb_df = pd.DataFrame(emb_ar, index=indices).add_suffix("_"+data.name)
        logger.debug("creating embeddings - finished")

        return emb_df
