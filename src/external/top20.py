"""Ermittle Zielwerte anhand der top 20%."""

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import Affine2D


def main():
    """Hauptfunktion."""
    # Daten einlesen
    data = read_file("Rohdaten.csv")
    aufbereiten(data)
    label = read_file("label.csv")
    zielwerte_labor = read_file("zielwerte_labor.csv")

    # Zielwerte berechnen und plotten
    zielwerte = get_top20(data, label[:18])
    write_file(zielwerte, "top20/zielwerte_top20.csv")
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


def plot_zielwerte(zielwerte: pd.DataFrame, zielwerte_labor: pd.DataFrame):
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
        "top20/",
    )


def plot_all_stadien(data: pd.DataFrame, label: str, path: str):
    all_kultur = data["Kultur"].unique()
    for kultur in all_kultur:
        Path(path + kultur).mkdir(parents=True, exist_ok=True)
        data_kultur = data[data["Kultur"] == kultur]
        all_nutrients = data_kultur["id_element"].unique()
        for element in all_nutrients:
            try:
                plot_stadien(data_kultur, kultur, element, label, path + kultur)
            except BaseException as e:
                print(kultur, element, e)
        plt.close("all")


def plot_stadien(
    data_kultur: pd.DataFrame, kultur: str, element: str, label: str, path: str
):
    stadien = data_kultur["Entwicklungsstadium"].unique()
    data_kultur = data_kultur[data_kultur["id_element"] == element]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.grid()
    ax.fill_between(
        stadien,
        data_kultur["min_labor"],
        data_kultur["max_labor"],
        label=fill(f"Literatur-Zielwertbereich für $NUTRIENT_FIXME in {kultur} (Bergmann, 1993; Vielemeyer und Hundt, 1991)", 60),
        color="blue",
        alpha=0.15,
    )
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
    ax.errorbar(
        stadien,
        data_kultur["mean_top"],
        yerr=data_kultur["std_top"],
        fmt="o",
        capsize=5,
        label=label,
        color="green",
        elinewidth=1,
        transform=trans1,
    )
    ax.errorbar(
        stadien,
        data_kultur["mean"],
        yerr=data_kultur["std"],
        fmt="o",
        capsize=5,
        label="Gesamtpopulation (alle Messungen) Mittelwert und Standardabweichung",
        color="blue",
        elinewidth=1,
        transform=trans2,
    )

    # Setze Titel und Achsenbeschriftung
    element_name = data_kultur["Element"].iloc[0]
    if isinstance(element_name, str):
        if " " in element_name:
            element_name = element_name[: element_name.find(" ")]
    else:
        element_name = data_kultur["id_element"].iloc[0]
    ax.set(
        title=f"{kultur} {element_name}",
        xlabel="Entwicklungsstadium",
        ylabel=f"{element_name} in {data_kultur['Einheit'].iloc[0]}",
    )
    ax.legend()

    # Speicher Diagramm
    fig.savefig(
        f"{path}/Zielwerte_{kultur}_{element_name}.png",
        transparent=False,
        dpi=300,
        bbox_inches="tight",
    )
    fig.clear()


def read_file(file_name: str):
    """Read csv."""
    return pd.read_csv(file_name, decimal=",", index_col=0)


def write_file(data: pd.DataFrame, file_name: str):
    """Write csv."""
    data.to_csv(file_name, sep=";", decimal=",", encoding="windows-1252", index=False)


if __name__ == "__main__":
    main()
