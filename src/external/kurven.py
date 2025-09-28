"""Ermittle Zielwerte anhand von Hüllkurven."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def main():
    """Hauptfunktion."""
    # Daten einlesen
    label = read_file("label.csv")
    data = read_file("Rohdaten.csv")
    # Hohe Konzentrationen raus werfen
    remove_high_values(data, label)
    # Kurven berechnen
    curves = calc_curves(data, label)
    write_file(curves, "kurven/Parameter.csv")
    plot_curves(curves, data)


def remove_high_values(data: pd.DataFrame, label: pd.DataFrame):
    """Alles größer 4 std-Abweichungen rauswerfen."""
    for kultur in data["kultur"].unique():
        for col, _ in label.iterrows():
            data_kultur = data[data["kultur"] == kultur][col]
            try:
                max_wert = data_kultur.mean() + 4 * data_kultur.std()
                data.loc[(data[col] > max_wert) & (data["kultur"] == kultur), col] = (
                    np.nan
                )
            except Exception:
                print(kultur, col)


def calc_curves(data: pd.DataFrame, label: pd.DataFrame):
    """Ermittle Hüllkurven."""
    rows = []
    # Filter auf Daten einer Kultur
    for kultur in data["kultur"].unique():
        data_kultur: pd.DataFrame = data[data["kultur"] == kultur]
        # Filter auf Daten eines Elements
        for col, row in label.iterrows():
            data_all = data_kultur[["ertrag (dt/ha)", col]].dropna()
            if data_all.empty:
                print(f"Keine Daten für {kultur} {col}")
                continue
            try:
                parameter = fit_curve(data_all.to_numpy())
                rows.append([kultur, col, row["name"]] + parameter)
            except Exception:
                print(f"Fehler bei {kultur} {col}")
    columns = ["Kultur", "id_element", "Variable", "y_max", "x_max", "a_l", "a_r"]
    curves = pd.DataFrame(rows, columns=columns)
    return curves


def fit_curve(data: np.ndarray):
    """Berechne Parameter eines Parabelsplines."""
    x, y = data[:, 1], data[:, 0]
    a_max = -np.max(y) / (np.max(x) - np.min(x)) ** 2
    par_start = [np.max(y), np.median(x), a_max, a_max]
    b_low = [0.6 * np.max(y), np.min(x), a_max, a_max]
    b_up = [1.2 * np.max(y), np.max(x), 0, 0]
    result = least_squares(error_spline, par_start, args=(x, y), bounds=(b_low, b_up))
    # p_opt = result.x
    # print(par_start, p_opt)
    # print(b_low, b_up)
    # print(f"iter: {result.nfev}, {result.message}\n")
    return list(np.round(result.x, decimals=8))


def error_spline(par, x, y):
    """Berechne Fehler eines Parabelsplines."""
    max_error = (max(y) - min(y)) / 2
    error = spline(x, *par) - y
    # error_under = np.sum(np.maximum(0, error))
    error_under = np.sum(np.minimum(np.maximum(0, error), max_error))
    # k = 0.5 * (max(y) - min(y))
    # error_above = k * np.sum(y > spline(x, *par))
    # error_above = 15 * np.sum(np.maximum(0, -error))
    # error_above = 20 * np.sum(np.minimum(np.maximum(0, -error), max_error)
    error_above = 6 * np.sum(np.minimum(np.maximum(0, -error), max_error))
    return error_under + error_above


def spline(input: np.ndarray, y_max, x_max, a_l, a_r):
    """Berechne Werte eines Parabelsplines."""
    return y_max + np.where(input < x_max, a_l, a_r) * (input - x_max) ** 2


def plot_curves(curves: pd.DataFrame, data: np.ndarray):
    """Plot Hüllkurven."""
    for _, curve in curves.iterrows():
        # Daten filtern
        kultur = curve["Kultur"]
        data_kultur = data[data["kultur"] == kultur]
        data_plot = (
            data_kultur[["ertrag (dt/ha)", curve["id_element"]]].dropna().to_numpy()
        )

        # Diagramm erstellen
        fig, ax = plt.subplots(figsize=(9, 6))
        x, y = data_plot[:, 1], data_plot[:, 0]
        ax.scatter(x, y, label="Kultur")
        x_spline = np.linspace(np.min(x), np.max(x), 100)
        y_spline = spline(
            x_spline, curve["y_max"], curve["x_max"], curve["a_l"], curve["a_r"]
        )
        ax.plot(x_spline, y_spline, label="Hüllkurve", color="green")
        # ax.legend()
        ax.set(
            title=f"{kultur}",
            xlabel=curve["Variable"],
            ylabel="Ertrag in dt/ha",
        )

        # Diagramm speichern
        Path(f"kurven/{kultur}").mkdir(parents=True, exist_ok=True)
        fig.savefig(
            f"kurven/{kultur}/Hüllkurve_{kultur}_{curve['id_element']}.png",
            transparent=False,
            dpi=300,
            bbox_inches="tight",
        )
        fig.clear()
        plt.close()


def read_file(file_name: str):
    """Read csv."""
    return pd.read_csv(file_name, decimal=",", index_col=0)


def write_file(data: pd.DataFrame, file_name: str):
    """Write csv."""
    data.to_csv(file_name, sep=";", decimal=",", encoding="windows-1252", index=False)


if __name__ == "__main__":
    main()
