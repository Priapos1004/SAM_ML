import pandas as pd
from pkg_resources import resource_filename

def get_plz_mapping() -> pd.DataFrame:
    """
    get dataframe with ort-postleitzahl-landkreis-bundesland mapping
    """
    filepath = resource_filename(__name__, 'zuordnung_plz_ort.csv')
    df = pd.read_csv(filepath)
    df =  df[["ort", "plz", "landkreis", "bundesland"]]
    return df
