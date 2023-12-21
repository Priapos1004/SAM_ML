import concurrent.futures
from typing import Literal

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm.auto import tqdm

from sam_ml.config import setup_logger

from ..main_data import Data

logger = setup_logger(__name__)


class Embeddings_builder(Data):
    """ Vectorizer Wrapper class """

    def __init__(self, algorithm: Literal["bert", "count", "tfidf"] = "tfidf", **kwargs):
        """
        Parameters
        ----------
        algorithm : {"bert", "count", "tfidf"}, \
                default="tfidf
            which vectorizing algorithm to use:
            - 'count': CountVectorizer (default)
            - 'tfidf': TfidfVectorizer
            - 'bert': SentenceTransformer("quora-distilbert-multilingual")

        \*\*kwargs:
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
        Function to get the possible parameter values for the class

        Returns
        -------
        param : dict
            possible values for the parameter "algorithm"

        Examples
        --------
        >>> # get possible parameters
        >>> from sam_ml.data.preprocessing import Embeddings_builder
        >>>
        >>> # first way without class object
        >>> params1 = Embeddings_builder.params()
        >>> print(params1)
        {"algorithm": ["tfidf", ...]}
        >>> # second way with class object
        >>> model = Embeddings_builder()
        >>> params2 = model.params()
        >>> print(params2)
        {"algorithm": ["tfidf", ...]}
        """
        param = {"algorithm": ["bert", "count", "tfidf"]}
        return param

    def get_params(self, deep: bool = True) -> dict:
        class_params = {"algorithm": self.algorithm}
        if self.algorithm != "bert":
            return class_params | self.transformer.get_params(deep)
        return class_params

    def set_params(self, **params):
        if self.algorithm == "bert":
            self._transformer = SentenceTransformer("quora-distilbert-multilingual", **params)
        else:
            self._transformer.set_params(**params)
        return self
    
    def create_parallel_bert_embeddings(self, content: list[str]) -> list:
        """
        Function to create in parallel embeddings of given strings with bert model

        Parameters
        ----------
        content : list[str]
            list of strings that shall be embedded

        Returns
        -------
        content_embeddings : list
            list of embedding vectors from content strings
        """
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
        Function to vectorize text data column

        Parameters
        ----------
        data : pd.Series
            column with text to vectorize
        train_on : bool, \
                default=True
            If ``True``, the estimator instance will be trained to build embeddings and then vectorize. Otherwise, it uses the trained instance for vectorizing.
        
        Returns
        -------
        emb_df : pd.DataFrame
            pandas Dataframe with vectorized data

        Examples
        --------
        >>> import pandas as pd
        >>> x_train = pd.Series(["Hallo world!", "Goodbye Island", "Greetings Berlin"], name="text")
        >>> x_test = pd.Series(["Goodbye world!", "Greetings Island"], name="text")
        >>> 
        >>> # vectorize data
        >>> from sam_ml.data.preprocessing import Embeddings_builder
        >>> 
        >>> model = Embeddings_builder()
        >>> x_train = model.vectorize(x_train) # train vectorizer
        >>> x_test = model.vectorize(x_test, train_on=False) # vectorize test data
        >>> print("x_train:")
        >>> print(x_train)
        >>> print()
        >>> print("x_test:")
        >>> print(x_test)
        x_train:
            0_text    1_text    2_text    3_text    4_text    5_text
        0   0.000000  0.000000  0.000000  0.707107  0.000000  0.707107
        1   0.000000  0.707107  0.000000  0.000000  0.707107  0.000000
        2   0.707107  0.000000  0.707107  0.000000  0.000000  0.000000
        <BLANKLINE>
        x_test:
            0_text  1_text    2_text    3_text  4_text    5_text
        0   0.0     0.707107  0.000000  0.0     0.000000  0.707107
        1   0.0     0.000000  0.707107  0.0     0.707107  0.000000
        """
        indices = data.index
        logger.debug("creating embeddings - started")
        if self.algorithm == "bert":
            message_embeddings = self.create_parallel_bert_embeddings(list(data))
            emb_ar = np.asarray(message_embeddings)

        else:
            if train_on:
                emb_ar = self._transformer.fit_transform(data).toarray()
            else:
                emb_ar = self._transformer.transform(data).toarray()

        emb_df = pd.DataFrame(emb_ar, index=indices).add_suffix("_"+data.name)
        logger.debug("creating embeddings - finished")

        return emb_df
