"""Ermittle Zielwerte anhand der top 20%."""

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import Affine2D
from anaplant import NUTRIENT_INFO, read_file
import numpy as np


def aufbereiten(data):
    """Normalisiere den Ertrag und benenne Kulturen um."""
    # Normalisiere den Ertrag
    ertrag = data.groupby("kultur")["ertrag (dt/ha)"]
    data["norm_ert"] = data["ertrag (dt/ha)"] / (
        ertrag.transform("max") * (1 + 0.5 / ertrag.transform("count"))
    )
    data.loc[data["kultur"] == "Körnererbse", "kultur"] = "Erbse"

def get_top20(data: pd.DataFrame, label: pd.DataFrame):
    """Ermittle Zielwerte anhand der Top 20%."""
    rows = []
    # Filter auf Daten einer Kultur
    for kultur in data["kultur"].unique():
        data_kultur: pd.DataFrame = data[data["kultur"] == kultur]
        # Filter auf Daten eines Elements
        for col, row in label.iterrows():
            data_all = data_kultur[["norm_ert", col]].dropna()
            # Berechne Zielwert für alle Entwicklungsstadien
            rows.append(calc_zielwert(data_all, kultur, "gesamt", col, row["name"]))
            # Berechne Zielwert für je Entwicklungsstadium
            for stadium in data_kultur["entwicklungsstadium"].unique():
                data_stadium = data_kultur[
                    data_kultur["entwicklungsstadium"] == stadium
                ]
                rows.append(
                    calc_zielwert(data_stadium, kultur, stadium, col, row["name"])
                )
    columns = [
        "Kultur",
        "Entwicklungsstadium",
        "id_element",
        "Variable",
        "Anzahl_top",
        "min_top",
        "max_top",
        "mean_top",
        "std_top",
        "Anzahl",
        "min",
        "max",
        "mean",
        "std",
    ]
    zielwerte = pd.DataFrame(rows, columns=columns)
    return zielwerte[zielwerte["Anzahl"] != 0]


def calc_zielwert(
    data_stadium: pd.DataFrame, kultur: str, stadium: str, col: str, name: str
):
    """Berechne den Zielwert für eine Kombination."""
    data_all = data_stadium[col]
    n = round(len(data_stadium) * 0.2)
    data_top = data_stadium.nlargest(n, "norm_ert")[col]

    return [
        kultur,
        stadium,
        col,
        name,
        data_top.count(),
        data_top.min(),
        data_top.max(),
        round(data_top.mean(), 4),
        round(data_top.std(), 4),
        data_all.count(),
        data_all.min(),
        data_all.max(),
        round(data_all.mean(), 4),
        round(data_all.std(), 4),
    ]


def plot_zielwerte(zielwerte: pd.DataFrame, zielwerte_labor: pd.DataFrame, path: str):
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
    plot_all_stadien(
        zielwerte_all,
        "Hochertragspopulation (20% höchste Erträge) Mittelwert und Standardabweichung",
        path,
    )


def plot_all_stadien(data: pd.DataFrame, label: str, path: str):
    all_kultur = data["Kultur"].unique()
    for kultur in all_kultur:
        data_kultur = data[data["Kultur"] == kultur]
        all_nutrients = data_kultur["id_element"].unique()
        for element in all_nutrients:
            try:
                plot_stadien(data_kultur, kultur, element, label, path)
            except Exception as e:
                print(kultur, element, e)
        plt.close("all")


def plot_stadien(
    data_kultur: pd.DataFrame, kultur: str, element: str, label: str, path: str
):
    stadien = data_kultur["Entwicklungsstadium"].unique()
    data_kultur = data_kultur[data_kultur["id_element"] == element]
    element_name, _, unit = NUTRIENT_INFO[element]
    fig, ax = plt.subplots(figsize=(9, 6))
    x_mask = ~np.isnan(data_kultur["min_labor"])
    ax.grid()

    ax.fill_between(
        stadien[x_mask],
        data_kultur["min_labor"][x_mask],
        data_kultur["max_labor"][x_mask],
        label=fill(f"Literatur-Zielwertbereich für {element_name} in {kultur} (Bergmann, 1993; Vielemeyer und Hundt, 1991)", 60),
        color="blue",
        alpha=0.15,
    )
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
    ax.errorbar(
        stadien[x_mask],
        data_kultur["mean_top"][x_mask],
        yerr=data_kultur["std_top"][x_mask],
        fmt="o",
        capsize=5,
        label=label,
        color="green",
        elinewidth=1,
        transform=trans1,
    )
    ax.errorbar(
        stadien[x_mask],
        data_kultur["mean"][x_mask],
        yerr=data_kultur["std"][x_mask],
        fmt="o",
        capsize=5,
        label="Gesamtpopulation (alle Messungen) Mittelwert und Standardabweichung",
        color="blue",
        elinewidth=1,
        transform=trans2,
    )

    ax.set(
        title=f"{kultur} {element_name}",
        xlabel="Entwicklungsstadium",
        ylabel=f"{element_name} in {unit}",
    )
    ax.legend()

    # Speicher Diagramm
    fig.savefig(
        f"{path}/Zielwerte_{kultur}_{element_name}.png".lower(),
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
