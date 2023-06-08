import pandas as pd
from datetime import datetime

from dask.distributed import Client
import dask.dataframe as dd

PATH_TO_PARKING_DATA = "/mnt/c/Users/Yannik/Downloads/Parking_Violations_Issued_-_Fiscal_Year_2023.csv"
types = {
    'Summons Number': 'int32',
    'Plate ID': 'string',
    'Registration State': 'string',
    'Plate Type': 'string',
    'Issue Date': 'string',
    'Violation Code': 'int16',
    'Vehicle Body Type': 'string',
    'Vehicle Make': 'string',
    'Issuing Agency': 'string',
    'Street Code1': 'int32',
    'Street Code2': 'int32',
    'Street Code3': 'int32',
    'Vehicle Expiration Date': 'int32',
    'Violation Location': 'string',
    'Violation Precinct': 'int32',
    'Issuer Precinct': 'int32',
    'Issuer Code': 'int32',
    'Issuer Command': 'string',
    'Issuer Squad': 'string',
    'Violation Time': 'string',
    'Time First Observed': 'string',
    'Violation County': 'string',
    'Violation In Front Of Or Opposite': 'string',
    'House Number': 'string',
    'Street Name': 'string',
    'Intersecting Street': 'string',
    'Date First Observed': 'int32',
    'Law Section': 'int32',
    'Sub Division': 'string',
    'Violation Legal Code': 'string',
    'Days Parking In Effect': 'string',
    'From Hours In Effect': 'string',
    'To Hours In Effect': 'string',
    'Vehicle Color': 'string',
    'Unregistered Vehicle?': 'string',
    'Vehicle Year': 'int16',
    'Meter Number': 'string',
    'Feet From Curb': 'int16',
    'Violation Post Code': 'string',
    'Violation Description': 'string',
    'No Standing or Stopping Violation': 'string',
    'Hydrant Violation': 'string',
    'Double Parking Violation': 'string'
}

def extract():
    df = dd.read_csv(PATH_TO_PARKING_DATA,\
                dtype=types, blocksize='100e6')
    
    df = df.rename(columns=lambda x: x.lower().rstrip().replace(' ', '_'))
    
    df['issue_date'] = dd.to_datetime(df['issue_date'])
    
    return df

def clean(df):
    # Handle null values
    df = df.dropna(how='all')
    
    # Drop unneeded columns
    
    # Detect outliers
    
    return df
    
    
    
def transform(df):
    # Find outliers
    # Add features: Year, Month, Day, Hour, Minute
    pass
    
def load(df, as_type='parquet'):
    if as_type == 'parquet':
        df.to_parquet('../data/1_PREPARED/parking_ticket_2023.parquet')
    
def main():
    client = Client(n_workers=3, threads_per_worker = 2, memory_limit='4G')
    #display(client)
    df = extract()
    df = clean(df)
    load(df)
    
if __name__ == "__main__":
    main()