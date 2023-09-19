import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pkg_resources import resource_filename

from .mapping import get_coord_main_cities

plt.style.use('seaborn-v0_8')

def visualise_plz(plz_region_df: pd.DataFrame, plot_col_name: str, plot_path: str = "german_map.png", plot_title: str = 'Germany map'):
    """
    @param:
        plz_region_df: dataframe with 'plz' column dtype string
        plot_col_name: column to plot
        plot_path: path for saving plot

    @soruce: https://www.suche-postleitzahl.org/downloads, 18/07/2023, Genauigkeit: mittel
    """
    filepath = resource_filename(__name__, 'plz-5stellig.shp')
    plz_shape_df = gpd.read_file(filepath, dtype={'plz': str})
    top_cities = get_coord_main_cities()

    germany_df = pd.merge(
        left=plz_shape_df, 
        right=plz_region_df, 
        on='plz',
        how='inner'
    )
    germany_df.drop(['note'], axis=1, inplace=True)

    plt.rcParams['figure.figsize'] = [16, 11]

    fig, ax = plt.subplots()

    germany_df.plot(
        ax=ax, 
        column=plot_col_name, 
        categorical=True, 
        legend=True, 
        legend_kwds={'title': plot_col_name, 'bbox_to_anchor': (1.35, 0.8)},
        cmap='tab20',
        alpha=0.9,
    )

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
    );

    # if default path -> add column name of plot column
    if plot_path == "german_map.png":
        plot_path = plot_path.split(".")[0]+f"_{plot_col_name}."+plot_path.split(".")[1]

    fig.savefig(plot_path)
