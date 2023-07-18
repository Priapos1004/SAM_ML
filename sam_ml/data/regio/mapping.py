import pandas as pd
from pkg_resources import resource_filename


def get_plz_mapping() -> pd.DataFrame:
    """
    get dataframe with ort-postleitzahl-landkreis-bundesland mapping

    @soruce: https://www.suche-postleitzahl.org/downloads, 18/07/2023
    """
    filepath = resource_filename(__name__, 'zuordnung_plz_ort.csv')
    df = pd.read_csv(filepath, dtype={'plz': str})
    df =  df[["ort", "plz", "landkreis", "bundesland"]]
    return df

def get_coord_main_cities() -> dict:
    """
    get coordinates of top cities from germany
    """
    top_cities = {
        'Berlin': (13.404954, 52.520008), 
        'Cologne': (6.953101, 50.935173),
        'Düsseldorf': (6.782048, 51.227144),
        'Frankfurt am Main': (8.682127, 50.110924),
        'Hamburg': (9.993682, 53.551086),
        'Leipzig': (12.387772, 51.343479),
        'Munich': (11.576124, 48.137154),
        'Dortmund': (7.468554, 51.513400),
        'Stuttgart': (9.181332, 48.777128),
        'Nuremberg': (11.077438, 49.449820),
        'Hannover': (9.73322, 52.37052)
    }
    return top_cities
