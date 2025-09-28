"""Ermittle Zielwerte anhand der top 20%."""

from pathlib import Path

from matplotlib import ticker
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import Affine2D
import polars as pl
from anaplant import NUTRIENT_INFO, NutrientInfo, read_file
import structlog
def main():
    """Hauptfunktion."""
    # Daten einlesen
    data = pl.read_excel("static/ANAPLANT Rohdaten-16-9-2024.xlsx").to_pandas()
    # data = read_file('static/ANAPLANT Rohdaten.xlsx')
    aufbereiten(data)
    label = read_file("external/label.csv")
    zielwerte_labor = read_file("external/zielwerte_labor.csv")

    # Zielwerte berechnen und plotten
    
    zielwerte = get_top20(data=data, label=label, nutrient_info=NUTRIENT_INFO)
    plot_zielwerte(zielwerte, zielwerte_labor)


def aufbereiten(data):
    """Normalisiere den Ertrag und benenne Kulturen um."""
    # Normalisiere den Ertrag
    ertrag = data.groupby("kultur")["ertrag (dt/ha)"]
    data["norm_ert"] = data["ertrag (dt/ha)"] / (
        ertrag.transform("max") * (1 + 0.5 / ertrag.transform("count"))
    )
    # Fasse Körnermais und Silomais mit relativen Erträgen zusammen
    data.loc[data["kultur"] == "Körnermais", "kultur"] = "Mais"
    data.loc[data["kultur"] == "Silomais", "kultur"] = "Mais"
    data.loc[data["kultur"] == "Körnererbse", "kultur"] = "Erbse"


def get_top20(
        *,
        data: pd.DataFrame, 
        label: pd.DataFrame,
        nutrient_info: dict[str, NutrientInfo]):
    """Ermittle Zielwerte anhand der Top 20%."""
    rows = []
    # Filter auf Daten einer Kultur
    for kultur in data["kultur"].unique():
        data_kultur: pd.DataFrame = data[data["kultur"] == kultur]
        # Filter auf Daten eines Elements
        for col, row in label.iterrows():
            data_all = data_kultur[["norm_ert", col, "jahr"]].dropna()
            # Berechne Zielwert für alle Entwicklungsstadien
            rows.append(
                calc_zielwert(
                    data_stadium=data_all, 
                    kultur=kultur, 
                    stadium="gesamt", 
                    col=col, 
                    name=row["name"]))
            # Berechne Zielwert für je Entwicklungsstadium
            for stadium in data_kultur["entwicklungsstadium"].unique():
                data_stadium = data_kultur[
                    data_kultur["entwicklungsstadium"] == stadium
                ]
                rows.append(
                    calc_zielwert(
                        data_stadium=data_stadium, 
                        kultur=kultur, 
                        stadium=stadium, 
                        col=col, 
                        name=row["name"])
                )
    columns = [
        "Kultur",
        "Entwicklungsstadium",
        "id_element",
        "Variable",
        "Anzahl",
        "mean_1",
        "std_1",
        "mean_2",
        "std_2",
        "mean_3",
        "std_3",
    ]
    zielwerte = pd.DataFrame(rows, columns=columns)
    return zielwerte[zielwerte["Anzahl"] != 0]


def calc_zielwert(
        *,
        data_stadium: pd.DataFrame, 
        kultur: str, 
        stadium: str, 
        col: str, 
        name: str
):
    """Berechne den Zielwert für eine Kombination."""
    data_1 = data_stadium[data_stadium["jahr"] == 1][col]
    data_2 = data_stadium[data_stadium["jahr"] == 2][col]
    data_3 = data_stadium[data_stadium["jahr"] == 3][col]
    return [
        kultur,
        stadium,
        col,
        name,
        data_stadium[col].count(),
        round(data_1.mean(), 4),
        round(data_1.std(), 4),
        round(data_2.mean(), 4),
        round(data_2.std(), 4),
        round(data_3.mean(), 4),
        round(data_3.std(), 4),
    ]


def plot_zielwerte(zielwerte: pd.DataFrame, zielwerte_labor: pd.DataFrame, plots_path: str):
    """Erzeuge ein Diagramm je Kultur und Element."""
    # Verknüpfe berechnete und Labor Zielwerte
    zielwerte_stadien = zielwerte[zielwerte["Entwicklungsstadium"] != "gesamt"]
    zielwerte_all = pd.merge(
        zielwerte_labor,
        zielwerte_stadien,
        how="outer",
        on=["Kultur", "Entwicklungsstadium", "id_element"],
    )
    # Erzeuge Diagramm
    plot_all_stadien(zielwerte_all, plots_path)


def plot_all_stadien(data: pd.DataFrame, path: str):
    log = structlog.get_logger()
    all_kultur = data["Kultur"].unique()
    for kultur in all_kultur:
        data_kultur = data[data["Kultur"] == kultur]
        all_nutrients = data_kultur["id_element"].unique()
        for element in all_nutrients:
            log = log.bind(nutrient=element, crop=kultur)
            try:
                plot_stadien(data_kultur, kultur, element, path)
            except (IndexError, TypeError) as e:
                log.exception(e)

from textwrap import fill
import numpy as np
def plot_stadien(data_kultur: pd.DataFrame, kultur: str, element: str, path: str):
    stadien = data_kultur["Entwicklungsstadium"].unique()
    data_kultur = data_kultur[data_kultur["id_element"] == element]
    element_name, _, unit =  NUTRIENT_INFO[element]
    

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.grid(alpha=.5)
    x_mask = ~np.isnan(data_kultur["min_labor"])
    ax.fill_between(
        stadien[x_mask],
        data_kultur["min_labor"][x_mask],
        data_kultur["max_labor"][x_mask],
        label=fill(f"Literatur-Zielwertbereich für {element_name} in {kultur} (Bergmann, 1993; Vielemeyer und Hundt, 1991)", 60),
        color="blue",
        alpha=0.15,
    )

    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(0.0, 0.0) + ax.transData
    trans3 = Affine2D().translate(+0.1, 0.0) + ax.transData
    
    ax.errorbar(
        stadien[x_mask],
        data_kultur["mean_1"][x_mask],
        yerr=data_kultur["std_1"][x_mask],
        fmt="o",
        capsize=5,
        label="Mittelwert und Standardabweichung, 2022",
        color="green",
        elinewidth=1,
        transform=trans1,
    )
    ax.errorbar(
        stadien[x_mask],
        data_kultur["mean_2"][x_mask],
        yerr=data_kultur["std_2"][x_mask],
        fmt="o",
        capsize=5,
        label="Mittelwert und Standardabweichung, 2023",
        color="blue",
        elinewidth=1,
        transform=trans2,
    )

    ax.errorbar(
        stadien[x_mask],
        data_kultur["mean_3"][x_mask],
        yerr=data_kultur["std_3"][x_mask],
        fmt="o",
        capsize=5,
        label="Mittelwert und Standardabweichung, 2024",
        color="r",
        elinewidth=1,
        transform=trans3,
    )

    # Setze Titel und Achsenbeschriftung

    ax.set(
        title=f"{kultur} {element_name}",
        xlabel="Entwicklungsstadium",
        ylabel=f"{element_name} in {unit}",
    )
    ax.legend()

    # Speicher Diagramm
    fig.savefig(
        f"{path}/zielwerte_{kultur}_{element_name}.png".lower(),
        transparent=False,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

def write_file(data: pd.DataFrame, file_name: str):
    """Write csv."""
    data.to_csv(file_name, sep=";", decimal=",", encoding="windows-1252", index=False)


if __name__ == "__main__":
    main()
