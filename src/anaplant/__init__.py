# SPDX-FileCopyrightText: 2024-present U.N. Owen <void@some.where>
#
# SPDX-License-Identifier: MIT
import re
from typing import TypeAlias
import polars as pl
import numpy as np
from datetime import datetime
import io
import pandas as pd

CUTOFF_DATE = datetime(2021, 6, 1)
NutrientInfo: TypeAlias = tuple[str, str, str]
NUTRIENT_INFO: dict[str, NutrientInfo] = {
    'p_n': ("Stickstoff", 'N', '% TS'),
    'p_c': ('Kohlenstoff', 'C',  '% TS'),
    'p_p': ('Phosphor', 'P', '% TS'),
    'p_k': ('Kalium', 'K', '% TS'),
    'p_ca': ('Calcium', 'Ca', '% TS'),
    'p_mg':('Magnesium', 'Mg', '% TS'),
    'p_na': ('Natrium', 'Na', '% TS'),
    'p_s':('Schwefel', 'S', '% TS'),
    'p_b':('Bor', 'B', 'ppm'),
    'p_mn':('Mangan', 'Mn', 'ppm'),
    'p_cu':('Kupfer', 'Cu', 'ppm'),
    'p_zn':('Zink', 'Zn', 'ppm'),
    'p_fe':('Eisen', 'Fe', 'ppm'),
    'p_mo':('Molybdän', 'Mo', 'ppm'),
    'p_al':('Aluminium', 'Al', 'ppm'),
    'p_co':('Kobalt', 'Co', 'ppm'),
    'p_se':('Selen', 'Se', 'ppm'),
    'p_c_n':("FILLER", 'C/N', 'Verhältnis'),
    'p_ts': ("FILLER", 'TS', '%'),
    'b_p':("Phosphor", 'P', 'mg/100 g'),
    'b_k':("Kalium", 'K', 'mg/100 g'),
    'b_mg':("Mangan", 'Mg', 'mg/100 g'),
    'b_ca':("Kalzium", 'Ca', 'mg/100 g'),
    'b_b':("Bor", 'B', 'mg/kg'),
    'b_mn':("Mangan", 'Mn', 'mg/kg'),
    'b_cu':("Kupfer", 'Cu', 'mg/kg'),
    'b_zn':("Zink", 'Zn', 'mg/kg'),
    'b_fe':("Eisen", 'Fe', 'mg/kg'),
    'b_c_n':("FILLER", 'C/N', ''),
    'b_c':("Kohlenstoff", 'C', '(% TS)2'),
    'b_n': ("Stickstoff", 'N', '% TS'),
    'b_humus': ('Humus', "Humus", '%TS')
}

station_read_schema = pl.Schema({
    "station_id":  pl.Int64,
    "start_date": pl.String,
    "end_date": pl.String,
    "elevation": pl.Float32,
    "lat": pl.Float32,
    "lon": pl.Float32,
    "station_name": pl.String,
    "station_state": pl.String})

def resave_german_weather_station_list(in_file: str, out_file: str):
    if out_file == in_file:
        raise ValueError('Out file is the same as in file. In-place modification is not allowed.')

    with open(in_file,  mode="r", encoding="UTF-8") as fh_in:
        with open(out_file, mode='w') as fh_out:
            for line_number, line in enumerate(fh_in):
                # skip first line
                if line_number == 0: 
                    columns = line.split(' ')
                    fh_out.write(','.join(columns))
                elif line.startswith('-'):
                    # skip the header / body delimeter
                    pass
                else:                
                    no_whitespace = tuple(filter(lambda v: v!= '', line.split(' ')))
                    station_id, from_date, to_date, station_height, geo_lat, geo_lon, *station_names, bundesland, newline = no_whitespace
                    names_joined = ' '.join(station_names)
                    station_name = f'"{names_joined}"'
                    fh_out.write(','.join([station_id, from_date, to_date, station_height, geo_lat, geo_lon, station_name, bundesland]) + newline)

def read_weather_station_csv(path: str) -> pl.DataFrame:
    station_df = pl.read_csv(path, schema=station_read_schema).with_columns(
        pl.col('start_date').str.to_date("%Y%m%d"),
        pl.col('end_date').str.to_date("%Y%m%d"))
    
    return station_df

def nearest_point(target: tuple[float, float], options: tuple[tuple[float, float]]):
    distances = np.zeros((len(options),), dtype='float')
    for idx, option in enumerate(options):
        distances[idx] = np.hypot(*np.subtract(target, option))
    return int(np.argmin(distances))

def add_nearest_station_column(rohdaten_df: pl.DataFrame, station_df: pl.DataFrame):
    # there is definitely a cleaner polars way to do this
    station_latlons = station_df.with_columns(lat_lon = pl.concat_list(pl.col('lat'), pl.col('lon')))['lat_lon'].to_list()
    rohdaten_latlons = rohdaten_df.with_columns(nearest_station=pl.concat_list(pl.col('gps_lat'), pl.col('gps_lon')))['nearest_station'].to_list()
    res = []
    for idx, r in enumerate(rohdaten_latlons):
        try: 
            pt_index = nearest_point(r, station_latlons)
            station_row = station_df[pt_index]
            station_id = station_row['station_id'].to_list()[0]
            station_name=station_row['station_name'].to_list()[0]
            station_lat = station_row['lat'].to_list()[0]
            station_lon = station_row['lon'].to_list()[0]
            res.append((station_id, station_name, station_lat, station_lon)) 
        except BaseException as e:
            print(idx, e)
            res.append((None,) * 4)

    ids, names, lats, lons = zip(*res)
    return rohdaten_df.with_columns(
        station_id=pl.Series(ids), 
        station_name=pl.Series(names),
        station_lat=pl.Series(lats),
        station_lon=pl.Series(lons))
    
def read_file(file_name: str, index_col: int = None) -> pd.DataFrame:
    """Read csv or excel file as pandas dataframe"""
    if file_name.endswith('.csv'):
        return pd.read_csv(file_name, decimal=",", index_col=index_col)
    elif file_name.endswith('.xlsx'):
        return pl.read_excel(file_name).to_pandas()
