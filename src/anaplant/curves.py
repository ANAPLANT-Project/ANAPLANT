"""
Ermittle Zielwerte anhand von Hüllkurven.
Originally written by Wulf Haberkern, with modifications by Davis Bennett
"""

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares      

def get_boundary_curve(
        *, 
        data: pd.DataFrame, 
        label: pd.DataFrame):

    """fit boundary curves"""
    rows = []

    for crop in filter(lambda v: isinstance(v, str), data["kultur"].unique()):
        data_kultur: pd.DataFrame = data[data["kultur"] == crop]

        # Filter auf Daten eines Elements
        for _, label_row in label.iterrows():
            row_id = label_row['id']
            row_name = label_row['name']
            data_yield = data_kultur[data_kultur[row_id + '_mask']][["ertrag (dt/ha)", row_id]].dropna()
            
            if data_yield.empty:
                print(f"Keine Daten für {crop} {row_id}")
                continue
            
            try:
                parameter = fit_curve(data_yield.to_numpy())
                rows.append([crop, row_id, row_name] + parameter)
            except BaseException as e:
                print(f"Fehler bei {crop} {row_id}: {e}")
    
    columns = ["kultur", "id_element", "Variable", "y_max", "x_max", "a_l", "a_r"]
    curves = pd.DataFrame(rows, columns=columns)
    return curves

def calc_curves(data: pd.DataFrame, label: pd.DataFrame):
    """Ermittle Hüllkurven."""
    rows = []
    # Filter auf Daten einer Kultur
    for kultur in data["kultur"].unique():
        data_kultur: pd.DataFrame = data[data["kultur"] == kultur]
        # Filter auf Daten eines Elements
        for _, label_row in label.iterrows():
            row_id = label_row['id']
            row_name = label_row['name']
            data_all = data_kultur[["ertrag (dt/ha)", row_id]].dropna()
            
            if data_all.empty:
                print(f"Keine Daten für {kultur} {row_id}")
                continue
            
            try:
                parameter = fit_curve(data_all.to_numpy())
                rows.append([kultur, row_id, row_name] + parameter)
            
            except Exception:
                print(f"Fehler bei {kultur} {row_id}")
    columns = ["kultur", "id_element", "Variable", "y_max", "x_max", "a_l", "a_r"]
    curves = pd.DataFrame(rows, columns=columns)
    return curves


def fit_curve(x: np.ndarray, y: np.ndarray):
    """Berechne Parameter eines Parabelsplines."""
    a_max = -np.max(y) / (np.max(x) - np.min(x)) ** 2
    par_start = [np.max(y), np.median(x), a_max, a_max]
    b_low = [0.6 * np.max(y), np.min(x), a_max, a_max]
    b_up = [1.2 * np.max(y), np.max(x), 0, 0]
    result = least_squares(error_spline, par_start, args=(x, y), bounds=(b_low, b_up))
    return list(np.round(result.x, decimals=8))


def error_spline(par, x, y):
    """Berechne Fehler eines Parabelsplines."""
    max_error = (max(y) - min(y)) / 2
    error = spline(x, *par) - y
    # error_under = np.sum(np.maximum(0, error))
    error_under = np.sum(np.minimum(np.maximum(0, error), max_error))
    error_above = 6 * np.sum(np.minimum(np.maximum(0, -error), max_error))
    return error_under + error_above


def spline(input: np.ndarray, y_max, x_max, a_l, a_r):
    """Berechne Werte eines Parabelsplines."""
    return y_max + np.where(input < x_max, a_l, a_r) * (input - x_max) ** 2

def percentile_threshold(
        data: np.ndarray, 
        *, 
        lower_percentile: float, 
        upper_percentile: float):
    """
    Return an array of bools where true indicates an outlier,
    false indicates an inlier. Thesholding is based on taking the values between the lower percentile 
    and the upper percentile. 
    """
    if np.any(np.isnan(data)):
        raise ValueError('Nans detected. Please remove them before calling this function.')
    threshold_lower, threshold_upper = np.percentile(data, [lower_percentile, upper_percentile])
    return np.logical_or(data < threshold_lower,  data > threshold_upper)


def get_nutrient_ranges(*, nutrient_range_data: pd.DataFrame, crop: str, nutrient: str) -> dict[str, tuple[float, float]]:
    nutrient_range = nutrient_range_data[np.logical_and(nutrient_range_data['Kultur'] == crop, nutrient_range_data['id_element'] == nutrient)]
    result = {}
    for _, row in nutrient_range.iterrows():
        result[row['Entwicklungsstadium']] = (row['min_labor'], row['max_labor'])
    return result

def plot_curves(
        *,
        crop_yield: np.ndarray,
        nutrient_conc: np.ndarray,
        measurement_site: np.ndarray,
        crop_name: str,
        nutrient_info: tuple[str,str, str],
        stages: tuple[str, ...],
        nutrient_range: tuple[float, float]
        ):
    
    if len(crop_yield) < 1:
        raise ValueError('Yield is empty.')

    yield_unit = 'dt/ha'
    text_width = 40

    nan_mask = np.logical_or(np.isnan(nutrient_conc), np.isnan(crop_yield))
    crop_yield_valid, nutrient_conc_valid = np.stack([crop_yield, nutrient_conc])[:, ~nan_mask]
    measurement_sites_valid = measurement_site[~nan_mask] 
    outlier_mask_y = percentile_threshold(crop_yield_valid, lower_percentile=0, upper_percentile=90)    
    outlier_mask_x = percentile_threshold(nutrient_conc_valid, lower_percentile=8, upper_percentile=92)
    outlier_masks_combined = np.logical_or(outlier_mask_y, outlier_mask_x)
    
    crop_yield_inliers, nutrient_conc_inliers = np.stack([crop_yield_valid, nutrient_conc_valid])[:, ~outlier_masks_combined]
    crop_yield_outliers, nutrient_conc_outliers = np.stack([crop_yield_valid, nutrient_conc_valid])[:, outlier_masks_combined]
    parameters = fit_curve(nutrient_conc_inliers, crop_yield_inliers)
    
    # Diagramm erstellen
    fig, ax = plt.subplots(figsize=(9, 6))
    # todo: break down the plotted data by origin 
    lab_sites = ()
    farm_sites = ()
    for maybe_lab_site in np.unique(measurement_sites_valid[:, 0]):
        example = np.where(measurement_sites_valid[:, 0] == maybe_lab_site)[0][0]
        if measurement_sites_valid[example][1] == 1:
            lab_sites += (maybe_lab_site,)
        else:
            farm_sites+= (maybe_lab_site,)  

    for index, site in enumerate(lab_sites + farm_sites):
        site_mask = measurement_sites_valid[:, 0] == site
        site_ort = measurement_sites_valid[site_mask][0][-1]
        if site in lab_sites:
            marker = lab_sites.index(site) + 5 
            label = f"{site}, {site_ort}"
        else:
            marker = 'o'
            if index == (len(lab_sites) + len(farm_sites)) - 1:
                label = "On-farm"
            else:
                label = None
        ax.scatter(nutrient_conc_valid[site_mask], crop_yield_valid[site_mask], label=label, marker=marker, color='gray')   
    ax.scatter(nutrient_conc_outliers, crop_yield_outliers, label="Ausreißer", marker='x', facecolor=None, s=100)

    x_spline = np.linspace(nutrient_conc_inliers.min(), nutrient_conc_inliers.max(), 100)
    y_spline = spline(
        x_spline, *parameters
    )
    yield_max = parameters[0]
    label = f'Literatur-Zielwertbereich für {nutrient_info[0]} in {crop_name} (Bergmann, 1993;  Vielemeyer und Hundt, 1991): {nutrient_range[0]:0.2f} - {nutrient_range[1]:0.2f} {nutrient_info[2]}'
    ax.plot(
        nutrient_range, 
        [yield_max * .95, yield_max * .95], 
        color='k', 
        linewidth=4, 
        linestyle='-',
        label=fill(label, text_width))
    # normalize to 0 and 1
    y_spline_normed = (y_spline - y_spline.min()) / (y_spline.max() - y_spline.min())

    try:
        new_range = x_spline[np.where(y_spline_normed > .9)[0][[0, -1]]]
    except IndexError:
        raise ValueError(f'Catastrophic curve fit, aborting plotting for {crop_name}, {nutrient_info[0]}')
    label = f'ANAPLANT-Zielwertbereich für {nutrient_info[0]} in {crop_name}, abgeleitet mit Hilfe einer Hüllkurve (Heym und Schnug, 1995, verändert): {new_range[0]:0.2f} - {new_range[1]:0.2f} {nutrient_info[2]}'
    ax.plot(
        new_range, 
        [yield_max * 1.05, yield_max * 1.05], 
        color='k', linewidth=4, linestyle=(0, (1,0.5)), label=fill(label, text_width))
    curve_legend = fill(f"Hüllkurve abgeleitet auf Basis der erhobene Daten. Ertrag max = {yield_max:0.2f} {yield_unit}", text_width)
    ax.plot(x_spline, y_spline, label=curve_legend, color="green")
    # ax.semilogx()
    ax.grid()
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set(
        title=f'{crop_name}\n ({stages})',
        xlabel=f"{nutrient_info[0]}gehalt in der Pflanze ({nutrient_info[2]})",
        ylabel=f"Ertrag in {yield_unit}",
    )
    plt.tight_layout()
    return fig, new_range


