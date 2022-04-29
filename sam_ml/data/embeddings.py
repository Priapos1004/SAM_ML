import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
    bert_active = True
except:
    print("build_embeddings(vec = 'bert') from data.bertembeddings cannot be used \n-> install 'sentence-transformers' to use this function")
    bert_active = False

from tqdm.notebook import tqdm


def build_embeddings(data: pd.DataFrame, vec: str = "count") -> pd.DataFrame:
    '''
    @param:
        data - DataFrame with columns that shall be encoded
        vec:
            'count' - CountVectorizer (default)
            'tfidf' - TfidfVectorizer
            'bert' - SentenceTransformer("quora-distilbert-multilingual")

    @return:
        DataFrame with encoded columns
    '''
    if bert_active and vec=='bert':
        print("- creating embeddings - using bert...")
        language_model = SentenceTransformer("quora-distilbert-multilingual")
        # Embedding creation
        message_embeddings = [language_model.encode(str(i)) for i in tqdm(data)]
        emb_ar = np.asarray(message_embeddings)

    elif vec=='count':
        print("- creating embeddings - using CountVectorizer...")
        cv=CountVectorizer(max_features=3000)
        emb_ar=cv.fit_transform(data).toarray()

    elif vec=='tfidf':
        print("- creating embeddings - using TfidfVectorizer...")
        tv=TfidfVectorizer(max_features=3000)
        emb_ar=tv.fit_transform(data).toarray()

    else:
        print(f"the entered vectorizer '{vec}' cannot be used --> using 'count' instead")
        print("- creating embeddings - using CountVectorizer...")
        cv=CountVectorizer(max_features=3000)
        emb_ar=cv.fit_transform(data).toarray()
        
    emb_df = pd.DataFrame(emb_ar)
    print("... embeddings created")

    return emb_df