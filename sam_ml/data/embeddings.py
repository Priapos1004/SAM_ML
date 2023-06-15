import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer

    bert_active = True
except:
    bert_active = False

from tqdm.auto import tqdm

from sam_ml.config import setup_logger

logger = setup_logger(__name__)


class Embeddings_builder:
    """ Vectorizer Wrapper class """

    def __init__(self, vec: str = "count", console_out: bool = False, **kwargs):
        """
        @param:
            vec:
                'count': CountVectorizer (default)
                'tfidf': TfidfVectorizer
                'bert': SentenceTransformer("quora-distilbert-multilingual")

            **kwargs:
                additional parameters for CountVectorizer or TfidfVectorizer
        """
        self.console_out = console_out
        self.vec_type = vec
        self._grid: dict[str, list] = {} # for pipeline structure

        if bert_active and vec == "bert":
            if self.console_out:
                logger.info("using quora-distilbert-multilingual model as vectorizer")
            self.vectorizer = SentenceTransformer("quora-distilbert-multilingual")

        elif not bert_active and vec == "bert":
            logger.warning("build_embeddings(vec = 'bert') from data.bertembeddings cannot be used \n-> install 'sentence-transformers' to use this function")

        elif vec == "count":
            if self.console_out:
                logger.info("using CountVectorizer as vectorizer")
            self.vectorizer = CountVectorizer(**kwargs)

        elif vec == "tfidf":
            if self.console_out:
                logger.info("using TfidfVectorizer as vectorizer")
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
        class_params = {"vec": self.vec_type, "console_out": self.console_out}
        if self.vec_type != "bert":
            return class_params | self.vectorizer.get_params(deep)
        return class_params | {"model_name_or_path": "quora-distilbert-multilingual"}

    def set_params(self, **params):
        if self.vec_type in ("bert"):
            self.vectorizer = SentenceTransformer("quora-distilbert-multilingual", **params)
        else:
            self.vectorizer.set_params(**params)
        return self

    def vectorize(self, data: pd.Series, train_on: bool = True) -> pd.DataFrame:
        """
        @params:
            data: pandas Series
            train_on: shall the vectorizer fit before transform
        @return:
            pandas Dataframe with vectorized data
        """
        indices = data.index
        if self.console_out:
            logger.debug("creating embeddings - started")
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
            logger.debug("creating embeddings - finished")

        return emb_df
