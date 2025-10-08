from itertools import product
from pathlib import Path
import click
from matplotlib import pyplot as plt
import numpy as np
from anaplant import NUTRIENT_INFO, read_file, resave_german_weather_station_list, read_weather_station_csv, add_nearest_station_column, CUTOFF_DATE
import polars as pl
import pandas as pd
import anaplant.curves as curves
import anaplant.top_percentile as top_percentile
import anaplant.years as years
import anaplant.apply_types as apply_types
from anaplant.util import decimal_comma_str_to_float

@click.group
def cli():
    pass

@click.command
@click.option('--source-path', type=click.STRING, required=True)
@click.option('--dest-path', type=click.STRING, required=True)
def resave_weather_station_list_cli(source_path: str, dest_path: str) -> None:
    resave_german_weather_station_list(source_path, dest_path)

@click.command
@click.option('--yield-data', type=click.STRING, required=True)
@click.option('--weather-station-list', type=click.STRING, required=True)
@click.option('--dest-path', type=click.STRING, required=True)

def localize_yields_cli(yield_data: str, weather_station_list: str, dest_path: str) -> None:
    rohdaten_df = pl.read_excel(yield_data)
    station_df = read_weather_station_csv(weather_station_list)

    result = add_nearest_station_column(
        rohdaten_df=rohdaten_df, 
        station_df=station_df.filter(pl.col('end_date') >= CUTOFF_DATE)
        )
    
    result.select(
    pl.col('station_id'), 
    pl.col('station_name'),
    pl.col('station_lat').cast(pl.String).map_elements(lambda v: v.replace('.', ',')),
    pl.col('station_lon').cast(pl.String).map_elements(lambda v: v.replace('.', ',')),
    ).write_csv(
        dest_path,
        separator=',',
        quote_char='"')

@click.command
@click.option('--yield-data', type=click.STRING, required=True)
@click.option('--nutrient-range-data', type=click.STRING, required=True)
@click.option('--crop', type=click.STRING, default=None, required=False)
@click.option('--plots-path', type=click.STRING, required=True)
@click.option('--nutrient', type=click.STRING, required=False)

def curves_cli(
    yield_data: str,
    nutrient_range_data: str, 
    crop: str | None,
    nutrient: str | None,
    plots_path: str) -> None:
    min_samples = 8

    yield_data_df = pl.read_csv(yield_data,
                                encoding='ISO8859-1',
                                separator=';',
                                infer_schema=False)
    yield_data_df = apply_types.types(yield_data_df)
    
    # combine entwicklungsstadiums
    yield_data_df = yield_data_df.with_columns(
        pl.col('entwicklungsstadium').replace('EC 64-65', 'EC 64')
    )
    range_schema = {'Kultur': pl.String, 'nutrient': pl.String, 'min': pl.Float64, 'max': pl.Float64}
    range_rows_out = []
    # combine Körnererbse and Erbse
    yield_data_df = yield_data_df.with_columns(pl.col('kultur').replace({'Körnererbse': 'Erbse'}))

    nutrient_range_data_df = pl.read_csv(nutrient_range_data)
        # duplicate Mais in Körnermais and Silomais
    mais = nutrient_range_data_df.filter(pl.col('Kultur') == 'Mais')
    kornermais = mais.with_columns(pl.col('Kultur').replace('Mais', 'Körnermais'))
    silomais = mais.with_columns(pl.col('Kultur').replace('Mais', 'Silomais'))
    nutrient_range_data_df = pl.concat([nutrient_range_data_df, kornermais, silomais])
    
    if crop is None:
        crops = tuple(yield_data_df['kultur'].unique().to_list())
    else:
        crops = (crop,)

    for _crop in crops:  

        if nutrient is None:
            nutrients = tuple(nutrient_range_data_df.filter(pl.col('Kultur') == _crop)['id_element'].to_list())
        else:
            nutrients = (nutrient,)

        for _nutrient in nutrients:
            nutrient_info =  NUTRIENT_INFO[_nutrient]
            nutrient_range_crop_element = nutrient_range_data_df.filter(
                (pl.col('Kultur') == _crop) & 
                (pl.col('id_element') == _nutrient)
            )
            
            # define equivalent classes of development stages
#            stage_map = {}
#            for row in nutrient_range_crop_element.iter_rows(named=True):
#                target_range = row['min_labor'], row['max_labor']
#                stage_name = row['Entwicklungsstadium']
#                if target_range not in stage_map:
#                    stage_map[target_range] = (stage_name, )
#                else:
#                    stage_map[target_range] = stage_map[target_range] + (stage_name, )
#
#            # fuse rows with equivalent min_labor and max_labor
            yield_by_crop = yield_data_df.filter(pl.col('kultur') == _crop)

#            def remap_stage(v):
#                for value in stage_map.values():
#                    if v in value:
#                        return value
#                return None
#            yield_by_crop = yield_by_crop.filter(pl.col('entwicklungsstadium').is_in(nutrient_range_crop_element['Entwicklungsstadium']))
#            yield_by_crop = yield_by_crop.with_columns(
#                pl.col('entwicklungsstadium').map_elements(
#                    remap_stage, return_dtype=pl.List(pl.String))).drop_nulls('entwicklungsstadium')

            for stages, data in yield_by_crop.group_by(pl.col('entwicklungsstadium')):
                nutrient_range = nutrient_range_data_df.filter(
                    (pl.col('Entwicklungsstadium') == stages[0]) & 
                    ((pl.col('id_element') == _nutrient)) & 
                    (pl.col('Kultur') == _crop)
                    )['min_labor', 'max_labor'].to_numpy().squeeze()

                mikro = _nutrient in ['p_b', 'p_mn', 'p_cu', 'p_zn', 'p_fe']
                
                crop_yield=data["ertrag (dt/ha)"].to_numpy()

                if len(crop_yield) >= min_samples:
                    try:
                        fig, new_range = curves.plot_curves(
                            crop_name=_crop, 
                            nutrient_info=nutrient_info, 
                            versuch=data['versuchsfläche'].to_numpy(),
                            oeko=data['öko/konv'].to_numpy(),
                            crop_yield=crop_yield,
                            nutrient_conc=data[_nutrient].to_numpy(),
                            stages=stages[0], 
                            nutrient_range=nutrient_range)
                        fname = Path(plots_path) / f'kurven_{_crop}_{nutrient_info[0]}_{stages[0]}'.lower()
                        if mikro:
                            fname = f'{fname}_gesamt'
                        fname = f'{fname}.png'
                        range_rows_out.append([_crop,_nutrient,stages[0], *new_range])
                        print(f'Saving {fname}\n')
                        fig.savefig(fname)
                        plt.close(fig)
                    except ValueError as e:
                        print(e.args)
                else:
                    msg = (
                        f'Not enough samples for crop {_crop}, nutrient {_nutrient}, stages {stages}.'
                        f'Got {len(crop_yield)} samples, needed {min_samples} or more.'
                        )
                    print(msg)

                if not mikro:
                    continue

                duengung_spalte = _nutrient.replace('p_', 'd_')

                data = data.remove(pl.col(duengung_spalte) & (pl.col('dat_düng') > pl.col('probenahme')))

                crop_yield=data["ertrag (dt/ha)"].to_numpy()

                if len(crop_yield) >= min_samples:
                    try:
                        fig, new_range = curves.plot_curves(
                            crop_name=_crop,
                            nutrient_info=nutrient_info,
                            versuch=data['versuchsfläche'].to_numpy(),
                            oeko=data['öko/konv'].to_numpy(),
                            crop_yield=crop_yield,
                            nutrient_conc=data[_nutrient].to_numpy(),
                            stages=stages[0],
                            nutrient_range=nutrient_range)
                        fname = Path(plots_path) / f'kurven_{_crop}_{nutrient_info[0]}_{stages[0]}'.lower()
                        fname = f'{fname}.png'
                        range_rows_out.append([_crop,_nutrient,stages[0], *new_range])
                        print(f'Saving {fname}\n')
                        fig.savefig(fname)
                        plt.close(fig)
                    except ValueError as e:
                        print(e.args)
                else:
                    msg = (
                        f'Not enough samples for crop {_crop}, nutrient {_nutrient}, stages {stages}.'
                        f'Got {len(crop_yield)} samples, needed {min_samples} or more.'
                        )
                    print(msg)

@click.command
@click.option('--yield-data', type=click.STRING, required=True)
@click.option('--nutrient-range-data', type=click.STRING, required=True)
@click.option('--plots-path', type=click.STRING, required=True)

def plot_top_percentile_cli(yield_data: str, plots_path: str, nutrient_range_data: str) -> None:
    data = read_file(yield_data)
    data.replace('EC 64-65', 'EC 64', inplace=True)
    top_percentile.aufbereiten(data)
    label = read_file("external/label.csv", index_col=0)
    zielwerte_labor = read_file(nutrient_range_data)
    # duplicate Mais in Körnermais and Silomais
    mais = (zielwerte_labor[zielwerte_labor['Kultur'] == "Mais"]).copy()
    kornermais = mais.copy().replace("Mais", "Körnermais")
    silomais = mais.copy().replace("Mais", "Silomais") 
    zielwerte_labor = pd.concat([zielwerte_labor,kornermais, silomais])
    zielwerte = top_percentile.get_top20(data, label)
    top_percentile.write_file(zielwerte, "external/top20/zielwerte_top20.csv")
    top_percentile.plot_zielwerte(zielwerte, zielwerte_labor, plots_path)

@click.command
@click.option('--yield-data', type=click.STRING, required=True)
@click.option('--plots-path', type=click.STRING, required=True)
@click.option('--nutrient-range-data', type=click.STRING, required=True)

def plot_annual_cli(yield_data: str, nutrient_range_data: str, plots_path: str) -> None:
    data = read_file(yield_data)
    data.replace('EC 64-65', 'EC 64', inplace=True)
    years.aufbereiten(data)
    label = read_file("external/label.csv", index_col=0)
    zielwerte_labor = read_file(nutrient_range_data)
    zielwerte = years.get_top20(data=data, label=label, nutrient_info=NUTRIENT_INFO)
    years.plot_zielwerte(zielwerte, zielwerte_labor, plots_path)

cli.add_command(resave_weather_station_list_cli, name='resave-weather-station-list')
cli.add_command(localize_yields_cli, name='localize-yields')
cli.add_command(curves_cli, 'plot-curves')
cli.add_command(plot_top_percentile_cli, 'plot-top-percentile')
cli.add_command(plot_annual_cli, 'plot-annual')

if __name__ == '__main__':
    cli()
