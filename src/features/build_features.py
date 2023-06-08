import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv

from dask.distributed import Client
import dask.dataframe as dd

from datetime import datetime

from meteostat import Stations, Daily, Point

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """Bulids dataset for high ticket day perdiction"""

    logger = logging.getLogger(__name__)
    logger.info('Build dataset for high ticket day prediction')

    client = Client(n_workers=3, threads_per_worker = 2, memory_limit='4G')

    ddf = dd.read_parquet(input_filepath)

    # Aggregate number of tickets per day
    ddf_aggregated = ddf[['summons_number','issue_date']].groupby(['issue_date']).count().compute().sort_index()

    # Set time period
    start = datetime(2022, 6, 1)
    end = datetime.now()

    newyork = Point(40.7143, -74.006, 57)

    # Get daily data
    data = Daily(newyork, start, end)
    weather_data = data.fetch()

    ddf_aggregated = ddf_aggregated.join(weather_data)

    ddf_aggregated.to_parquet(output_filepath)
    client.shutdown()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
