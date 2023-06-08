# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv

from dask.distributed import Client
import dask.dataframe as dd

from config import column_types


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('file_type', type=click.Choice(['PARQUET', 'HDFS', 'all']))
def main(input_filepath, output_filepath, file_type):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Transform raw parking data to processed file')

    client = Client(n_workers=3, threads_per_worker = 2, memory_limit='4G')

    # Read in original csv file
    df = dd.read_csv(input_filepath, dtype=column_types, blocksize=100e6)

    # Rename all columns to be lower case and connect words with '_'
    df = df.rename(columns=lambda x: x.lower().rstrip().replace(' ', '_'))

    # Set issue_date to datetime object
    df['issue_date'] = dd.to_datetime(df['issue_date'])

    if file_type == 'PARQUET':
        df.to_parquet(output_filepath+'.parquet')
    elif file_type == 'h5':
        df.to_hdf(output_filepath+'h5', key='parking_ticket')
    elif file_type == 'all':
        df.to_parquet(output_filepath+'.parquet')
        df.to_hdf(output_filepath+'h5', key='parking_ticket')
    else:
        raise Exception('No enabled file type give. Please define output_filepath to be either PARQUET or HDFS(h5) type')

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
