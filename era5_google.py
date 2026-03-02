import xarray as xr
import pandas as pd
from typing import Union

class Client:
    """
    Client for the google cloud era5 surface data. 
    This class contains methods for exploring and retrieving data.
    Requirements: installation of google gloud library gcsfs and zarr (and fsspec, or is this automatic?)
    """

    def __init__(self):
        print("Starting session...")
        self.data = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", 
        consolidated=True,
        chunks={'time': 48},
        )
    
    def available_variables(self, search_for : str | None = None) -> Union[dict,list]:
        """
        Returns overview of all data variables available for downloading 

        :param search_for: str, (part of) variable name to search for
        :return: either a dictionary of all available variables, or a list of variables with the search_for string within them 
        :Example:
            >>> client = Client()
            >>> variables = client.available_variables()
            >>> evap_vars = client.available_variables("evap")
        """
        vars = self.data.data_vars
        if search_for is None:
            return vars
        else: 
            vars_subset = []
            for var in vars:
                if search_for in var:
                    vars_subset.append(var)
            return vars_subset

    def get_data(self, variables: list, t_start: str, t_end: str, lon: list[float], lat: list[float], aggregate = None, save=False) -> xr.Dataset:
        """
        Retrieve a subset of the data for the given variables and time range
        Data is slices according to lon-lat coordinates
        Note that it will often be just as fast to download the entire range without slicing and computing the aggregate

        :param variables: list of variables to download. 
        :param t_start: start time in format 'YYYY-MM-DD'
        :param t_end: end time in format 'YYYY-MM-DD'
        :param lon: list of two floats representing the longitude range, needs to be in [-180, 180] [lonmin,lonmax]
        :param lat: list of two floats representing the latitude range, needs to be in [90,-90] [latmax,latmin] DESCENDING!
        :param aggregate: aggregation time period, e.g. '1MS' for monthly aggregation, None for no aggregation. Strings allowed follow xarray resample method
        :param save: bool, keeps lazy operations only if set to False, opens values if set to True
        :return: xarray dataset with the requested data

        :Example:
            >>> client = Client()
            >>> data = client.get_data(variables = ["2m_temperature"], t_start = '2020-01-01', t_end= '2020-12-31', lon = [-81, -68], lat = [-19, 1], aggregate = "1MS")

        """

        if lon[0]<-180 or lon[0]>180 or lon[1]<-180 or lon[1]>180:
             raise ValueError("Invalid coordinates: Lon must be between -180 and 180")
        if lat[0]<-90 or lat[0]>90 or lat[1]<-90 or lat[1]>90:
             raise ValueError("Invalid coordinates: Lat must be between -90 and 90")
        if lat[0] < lat[1]:
             raise ValueError("Invalid coordinates: Lat coordinates must be descending")
        if lat[0]-lat[1]<0.25:
             raise ValueError("Invalid coordinates: Box must be larger than 0.25")
        if lon[1]-lon[0]<0.25:
             raise ValueError("Invalid coordinates: Box must be larger than 0.25")
        

        print("Fetching and slicing data...")
        self.data = self.data.assign_coords(longitude=(((self.data.longitude + 180) % 360) - 180))
        self.data = self.data.reindex({"longitude": sorted(self.data.longitude)})
        data_sliced = self.data.sel(time=slice(t_start, t_end),latitude=slice(lat[0], lat[1]), longitude=slice(lon[0], lon[1]), level = 1)
        data_sliced = data_sliced[variables]
        
        if aggregate is not None:
            data_sliced = data_sliced.resample(time=aggregate, skipna = True).mean(dim="time")

        if save:
            ("Saving data to variable...")
            data_downloaded = data_sliced.compute()
            return data_downloaded
        return data_sliced

    def get_data_point(self, variables: list, t_start: str, t_end: str, lon: float, lat: float, aggregate = None, save=False) -> pd.DataFrame:
        """
        Retrieve a subset of the data for the given variables and time range
        Data is slices according to lon-lat coordinates
        Note that it will often be just as fast to download the entire range without slicing and computing the aggregate
        WARNING: This method seems incredibly slow

        :param variables: list of variables to download. 
        :param t_start: start time in format 'YYYY-MM-DD'
        :param t_end: end time in format 'YYYY-MM-DD'
        :param lon: float, needs to be in [-180, 180]
        :param lat: float, needs to be in [90,-90]
        :param aggregate: aggregation time period, e.g. '1MS' for monthly aggregation, None for no aggregation. Strings allowed follow xarray resample method
        :param save: bool, keeps lazy operations only if set to False, opens values if set to True
        :return: DataFrame with the requested data

        :Example:
            >>> client = Client()
            >>> data = client.get_data(variables = ["2m_temperature"], t_start = '2020-01-01', t_end= '2020-12-31', lon = [-81, -68], lat = [-19, 1], aggregate = "1MS")

        """
        print("Fetching and slicing data...")

        if lon<-180 or lon>180:
             raise ValueError("Invalid coordinates: Lon must be between -180 and 180")
        if lat<-90 or lat>90:
             raise ValueError("Invalid coordinates: Lat must be between -90 and 90")

        self.data = self.data.assign_coords(longitude=(((self.data.longitude + 180) % 360) - 180))
        self.data = self.data.reindex({"longitude": sorted(self.data.longitude)})
        data_sliced = self.data.sel(time=slice(t_start, t_end))
        data_sliced = data_sliced.sel(latitude=lat, longitude=lon, level = 1,method='nearest')
        data_sliced = data_sliced[variables]
        
        if aggregate is not None:
            data_sliced = data_sliced.resample(time=aggregate, skipna = True).mean(dim="time")

        if save:
            ("Saving data to variable...")
            data_downloaded = data_sliced.compute()
            return data_downloaded
        return data_sliced.to_dataframe().drop(['latitude', 'level', 'longitude'], axis=1)
