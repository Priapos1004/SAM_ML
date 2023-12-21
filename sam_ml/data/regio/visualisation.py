import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pkg_resources import resource_filename

from .mapping import get_coord_main_cities

plt.style.use('seaborn-v0_8')

def visualise_plz(plz_region_df: pd.DataFrame, plot_col_name: str, plot_path: str = "german_map.png", plot_title: str = "Germany map"):
    """
    Function to visualise data values mapped to zipcode on germany map

    Parameters
    ----------
    plz_region_df : pd.DataFrame
        dataframe with ``plz`` column dtype string
    plot_col_name : str
        column to plot
    plot_path : str, default="german_map.png"
        path for saving plot
    plot_title : str, default="Germany map"
        title of plot

    Returns
    -------
    saves plot at ``plot_path``. If default path, then the column name of plot column will be added

    Notes
    -----
    Source: https://www.suche-postleitzahl.org/downloads, 18/07/2023, Genauigkeit: mittel

    Examples
    --------

    First example with less than 8 different unique values to plot (label legend):

    >>> # load data (replace with own data)
    >>> import pandas as pd
    >>> df = pd.DataFrame({"plz": ['78267', '54298', '52062', '52064', '52066', '52068', '52070', '52072', '52074', '52076'], "income": [1400, 700, 2400, 1400, 300, 2400, 700, 1000, 1400, 2000]})
    >>>
    >>> from sam_ml.data.regio import visualise_plz
    >>> visualise_plz(df, plot_col_name="income")

    Second example with more than 8 different unique values to plot (colorbar legend):

    >>> # load data (replace with own data)
    >>> import pandas as pd
    >>> df = pd.DataFrame({"plz": ['78267', '54298', '52062', '52064', '52066', '52068', '52070', '52072', '52074', '52076'], "income": [1400, 800, 2400, 1400, 300, 2400, 700, 1000, 1600, 2000]})
    >>>
    >>> from sam_ml.data.regio import visualise_plz
    >>> visualise_plz(df, plot_col_name="income")
    """
    filepath = resource_filename(__name__, 'plz-5stellig.shp')
    plz_shape_df = gpd.read_file(filepath, dtype={'plz': str})
    top_cities = get_coord_main_cities()

    germany_df = pd.merge(
        left=plz_shape_df, 
        right=plz_region_df,
        on='plz',
        how='left'
    )
    germany_df.drop(['note'], axis=1, inplace=True)

    plt.rcParams['figure.figsize'] = [16, 11]

    fig, ax = plt.subplots()

    # discrete vs continuous data legend
    discrete_data = len(germany_df[plot_col_name].unique()) < 8

    plot = germany_df.plot(
        ax=ax, 
        column=plot_col_name,
        categorical=True,
        legend=discrete_data,
        legend_kwds={'title': plot_col_name, 'bbox_to_anchor': (1.35, 0.8)},
        cmap='jet',
        alpha=0.9,
        missing_kwds={
            "color": "lightgrey",  # Color for NaN values
            "label": "Missing data"  # Label for legend
        },
    )

    if not discrete_data:
        # Create a ScalarMappable with the same colormap and normalization as the plot
        norm = Normalize(vmin=germany_df[plot_col_name].min(), vmax=germany_df[plot_col_name].max())
        sm = ScalarMappable(cmap='jet', norm=norm)
        sm._A = []  # Empty array for the data range
        fig.colorbar(sm, ax=ax, label=plot_col_name)

    for c in top_cities.keys():

        ax.text(
            x=top_cities[c][0], 
            y=top_cities[c][1] + 0.08, 
            s=c, 
            fontsize=12,
            ha='center', 
        )

        ax.plot(
            top_cities[c][0], 
            top_cities[c][1], 
            marker='o',
            c='black', 
            alpha=0.5,
        )

    ax.set(
        title=plot_title,
        aspect=1.3,
        facecolor='white',
    )

    # if default path -> add column name of plot column
    if plot_path == "german_map.png":
        plot_path = plot_path.split(".")[0]+f"_{plot_col_name}."+plot_path.split(".")[1]

    fig.savefig(plot_path)
