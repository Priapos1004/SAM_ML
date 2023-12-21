import pandas as pd
from pkg_resources import resource_filename


def get_plz_mapping() -> pd.DataFrame:
    """
    Function to get dataframe with ort-postleitzahl-landkreis-bundesland mapping

    Returns
    -------
    df_mapping : pd.Dataframe
        dataframe with columns "ort", "plz", "landkreis", and "bundesland"

    Notes
    -----
    Source: https://www.suche-postleitzahl.org/downloads, "zuordnung_plz_ort.csv", 18/07/2023

    Examples
    --------
    >>> from sam_ml.data.regio import get_plz_mapping
    >>> get_plz_mapping()
        ort     plz     landkreis                   bundesland
    0   Aach    78267   Landkreis Konstanz          Baden-Württemberg
    1   Aach    54298   Landkreis Trier-Saarburg    Rheinland-Pfalz
    ...
    """
    filepath = resource_filename(__name__, 'zuordnung_plz_ort.csv')
    df_mapping = pd.read_csv(filepath, dtype={'plz': str})
    df_mapping =  df_mapping[["ort", "plz", "landkreis", "bundesland"]]
    return df_mapping

def get_coord_main_cities() -> dict[str, tuple[float, float]]:
    """
    Function to get coordinates of top cities from germany

    Returns
    -------
    top_cities : dict[str, tuple[float, float]]
        dictionary with english names of top german cities and their coordinates as values

    Examples
    --------
    >>> from sam_ml.data.regio import get_coord_main_cities
    >>> get_coord_main_cities()
    {...}
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
