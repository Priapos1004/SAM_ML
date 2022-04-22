'''
WARNING: This script does not work on M1 (state 22.04.2022) because sentence transformers is not supported
'''
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm


def build_embeddings(data: pd.DataFrame) -> pd.DataFrame:
    '''
    @param:
        data - DataFrame with columns that shall be encoded

    @return:
        DataFrame with encoded columns
    '''
    language_model = SentenceTransformer("quora-distilbert-multilingual")
    # Embedding creation
    print("- creating embeddings")
    message_embeddings = [language_model.encode(str(i)) for i in tqdm(data)]
    ar = np.asarray(message_embeddings)
    df_BERT = pd.DataFrame(ar)
    print("- embeddings created")
    
    return df_BERT