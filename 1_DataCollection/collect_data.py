import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly

def get_ticket_data():
    df = pd.read_csv('https://data.cityofnewyork.us/resource/7mxj-7a6y.csv')
    
    print(df.info())
    print(df.head(1))
    
    df.to_parquet('../data/0_RAW/parking_ticket_2022.parquet')
    df.to_hdf('../data/0_RAW/parking_ticket_2022.h5', key='parking_ticket')
    
def get_weather_data():
    # Set time period
    start = datetime(2022, 1, 1)
    end = datetime(2022, 12, 31)

    # Create Point for Vancouver, BC
    newyork = Point(40.7143, -74.006, 57)

    # Get daily data for 2018
    data = Hourly(newyork, start, end)
    data = data.fetch()
    
    print(data.info())
    print(data.head(2))
    
    data.to_parquet('../data/0_RAW/weather_2022.parquet')
    #data.to_hdf('0_RAW/weather_2022.h5', key='weather')
    
def get_school_data():
    data = pd.read_csv('https://data.cityofnewyork.us/resource/wg9x-4ke6.csv')
    
    print(data.info())
    print(data.head(1))
    
    data.to_parquet('../data/0_RAW/school_locations.parquet')
    data.to_hdf('../data/0_RAW/school_locations.h5', key='school_locations')
    
def get_event_data():
    pass

def get_major_attractions_data():
    '''
    https://data.cityofnewyork.us/City-Government/Points-Of-Interest/rxuy-2muj
    https://data.cityofnewyork.us/resource/t95h-5fsr.json
    '''
    data = pd.read_csv("https://data.cityofnewyork.us/resource/t95h-5fsr.csv")
    print(data.info())
    print(data.head(1))
    # need to convert the_geom column to location
def get_business_data():
    '''
    This data set features businesses/individuals holding a DCA license so that they may legally operate in New York City.
    Note: Sightseeing guides and temporary street fair vendors are not included in this data set.
    '''
    data = pd.read_csv("https://data.cityofnewyork.us/resource/w7w3-xahh.csv")

    print(data.info())
    print(data.head(1))
    
    data.to_parquet("../data/0_RAW/business_locations.parquet")
    data.to_hdf("../data/0_RAW/business_locations.h5", key="business_locations")
    
def get_attraction_data():
    # get data from overpass though open street map data
    """/*
    This query looks for nodes, ways and relations 
    with the given key/value combination.
    Choose your region and hit the Run button above!
    */
    [out:json][timeout:25];
    // gather results
    (
    // query part for: “tourism=attraction”
    node["tourism"="attraction"]({{bbox}});
    way["tourism"="attraction"]({{bbox}});
    relation["tourism"="attraction"]({{bbox}});
    );
    // print results
    out body;
    >;
    out skel qt;
    """
    pass

#get_ticket_data()
#get_weather_data()
#get_school_data()
#get_business_data()
get_major_attractions_data()