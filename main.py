# %% Cargar dependencias
from datetime import date
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

"""

Este script implementa un modelo de aprendizaje no supervisado basado
en K-Means para agrupar acciones según su perfil combinado de riesgo y
retorno. Se utilizan métricas históricas de precios para construir los
atributos del modelo.

REQUISITO: tener un archivo .xlsx con precios históricos o usar yf.
En una entrega anterior no me recomendaron usar yfinance porque "está
inestable", así que si bien voy a incluir una función yf.download, voy
a dejar la carga del archivo como responsabilidad del usuario mediante
un xlsx file con un bloque try except. Si no se encuentra el atchivo
xlsx, entonces se procederá a usar yfinance. En caso de usar un xlsx
local, este archivo tiene que tener un formato como el del siguiente
ejemplo:

Date	Stock1	Stock2	Stock3	Stock4
02-01-2020	73	68	125	120
03-01-2020	72	68	124	119
06-01-2020	72	69	123	118
07-01-2020	72	69	124	116
08-01-2020	73	70	124	117
09-01-2020	75	71	124	118

"""


def _get_stock_data(
    tickers: list[str], start: date, end: date
) -> pd.DataFrame | None:
    # Función helper completamente opcional. Baja retornos de yfinance.
    try:
        df_stocks = yf.download(
            tickers, start=start, end=end, auto_adjust=False
        )
    except Exception as e:
        print(f"Error al descargar datos de yfinance: {e}")
        return None
    if df_stocks is None:
        raise ValueError("Check df_stocks for problems.")
    try:
        prices = cast(pd.DataFrame, df_stocks["Adj Close"])
    except Exception as e:
        print(f"No se encontró la columna 'Adj Close'. Revisar datos: {e}")
    else:
        return prices


def calculate_features(prices: pd.DataFrame) -> pd.DataFrame:
    # Calcular métricas a usar en el clustering. Se usarán volatilidad y
    # retorno anualizados junto con maximum drawdown
    daily_returns = cast(
        pd.DataFrame, np.log(prices / prices.shift(1))
    ).dropna()

    features = pd.DataFrame(index=prices.columns)
    features["Volatilidad"] = daily_returns.std() * np.sqrt(252)
    features["Retorno_Promedio"] = daily_returns.mean() * 252
    features["Max_Drawdown"] = ((prices / prices.cummax()) - 1).min()

    return features


def find_best_k(X_scaled: np.ndarray, k_range: range) -> int:
    # Determina el número óptimo de clusters (K) utilizando el silhouette score
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        sil_scores.append(score)

    plt.plot(k_range, sil_scores, marker="o")
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Selección de K óptimo con Silhouette Score")
    plt.grid(True)
    plt.show()

    return k_range[np.argmax(sil_scores)]


def plot_clusters(features: pd.DataFrame) -> None:
    # Generar un gráfico de dispersión para poder visualizar los clusters
    sns.scatterplot(
        data=features,
        x="Volatilidad",
        y="Retorno_Promedio",
        hue="Cluster",
        palette="viridis",
        s=80,
    )
    plt.title("Clustering de acciones por perfil de riesgo-retorno")
    plt.grid(True)
    # plt.savefig("clusters_riesgo_retorno.png")
    # Lo usé una vez para guardar los clusters
    plt.show()


def main():
    # Función que ejecuta todo el proceso.
    tickers = [
        "TSLA",  # Tesla Inc. - Automotriz / Tecnología
        "AMD",  # Advanced Micro Devices - Semiconductores
        "SHOP",  # Shopify Inc. - E-commerce / Tecnología
        "PYPL",  # PayPal Holdings Inc. - Pagos digitales / Fintech
        "SO",  # Southern Company - Utilities / Eléctrica
        "PLTR",  # Palantir Technologies - Software / Big Data
        "RIVN",  # Rivian Automotive Inc. - Automotriz eléctrica
        "PG",  # Procter & Gamble Co. - Consumo básico / Hogar
        "BIDU",  # Baidu Inc. - Internet / IA
        "TDOC",  # Teladoc Health Inc. - Salud digital
        "KO",  # Coca-Cola Co. - Consumo básico / Bebidas
        "DE",  # Deere & Co. - Industriales / Maquinaria agrícola
        "JNJ",  # Johnson & Johnson - Salud / Farma / Consumo
        "PEP",  # PepsiCo Inc. - Consumo básico / Bebidas y snacks
        "ARKK",  # ARK Innovation ETF - ETF Disruptivo
        "XOM",  # Exxon Mobil Corp. - Energía / Petróleo
        "WMT",  # Walmart Inc. - Retail / Consumo masivo
        "NEE",  # NextEra Energy Inc. - Utilities / Energía renovable
        "MRK",  # Merck & Co. Inc. - Farma / Salud
        "XLP",  # Consumer Staples ETF - ETF de consumo básico
        "NVDA",  # NVIDIA Corp. - Semiconductores / IA
        "DUK",  # Duke Energy Corp. - Utilities / Eléctrica
        "CAT",  # Caterpillar Inc. - Industriales / Maquinaria pesada
        "CVX",  # Chevron Corp. - Energía / Petróleo
        "LMT",  # Lockheed Martin Corp. - Aeroespacial / Defensa
        "FDX",  # FedEx Corp. - Transporte / Logística
        "BA",  # Boeing Co. - Aeroespacial / Aviación
        "JPM",  # JPMorgan Chase & Co. - Finanzas / Banca
        "GS",  # Goldman Sachs Group Inc. - Finanzas / Inversión
        "UNP",  # Union Pacific Corp. - Transporte ferroviario
    ]

    # --- 1. Obtención de datos
    try:
        # Bloque xlsx
        prices = pd.read_excel("stock_data.xlsx")
        print("Se leyeron los datos del archivo excel...")
        if "Date" in prices.columns:
            prices.set_index("Date", inplace=True)
    except Exception:
        # Bloque yfinance
        start_date = date(2022, 1, 2)
        end_date = date(2024, 12, 30)

        print("Descargando datos de yfinance...")
        prices = _get_stock_data(tickers, start=start_date, end=end_date)
    if prices is None:
        print("Finalizando el script debido a un error en los datos.")
        raise Exception
    # prices.to_excel("stock_data.xlsx")
    # Usé el snippet anterior una sola vez para no estar bajando los datos de
    # yf cada vez que corro el código.
    # --- 2. Cálculo de métricas
    print("Calculando métricas de riesgo y retorno...")
    features = calculate_features(prices)

    # --- 3. Estandarizado de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # --- 4. Encontrar el mejor K
    print("Buscando el número óptimo de clusters (K)...")
    # Pongo un límite de 6 clusters por la cantidad de empresas bajo estudio
    k_range = range(2, 7)
    best_k = find_best_k(X_scaled, k_range)
    print(f"El número óptimo de clusters es: {best_k}")

    # --- 5. Entrenamiento del modelo final y asignación de clusters
    print("Entrenando el modelo final...")
    model = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    features["Cluster"] = model.fit_predict(X_scaled)

    # --- 6. Visualización y exportación
    print("Generando visualización de los clusters...")
    plot_clusters(features)

    # print("Exportando resultados a 'resultados_clusters.xlsx'...")
    # features.to_excel("resultados_clusters.xlsx")
    # Igual que en los casos anteriores, lo usé una vez para guardar todo.
    print("Proceso completado exitosamente.")

    """Clusters observados y conclusiones generales:
    Cluster 0: Tech Ganador 🚀 (NVDA y PLTR)
    Empresas Tech hiper agresivas.
    Ambas han tenido un tremendo crecimiento en los últimos años y se observa
    en la data que tienen retornos muy altos, pero también alta volatilidad y
    alto drawdown.
    Han aprovechado mucho el boom de la tecnología y la IA.

    Cluster 1: Tech Castigado 😢 (PYPL, SHOP, TSLA, etc.)
    También son empresas tecnológicas con muy alta volatilidad y drawdown,
    pero que no han logrado tener retornos tan exorbitantes como las
    anteriores. En la mayoría de los casos han sido negativos de hecho.
    Riesgo alto sin recompensa clara.

    Cluster 2: Defensivo Estable 🛡️ (CAT, JPM, WMT, GS, etc.)
    El cluster más grande. Aquí tenemos todo el resto de empresas con
    volatilidades mucho más estables que los dos grupos anteriores Los sectores
    son muy diversos, incluyendo financieras, consumo retail y maquinaria,
    entre otros. A pesar de esto, lo que tienen en común es su perfil riesgo
    retorno mucho más equilibrado.
    """


if __name__ == "__main__":
    from rich import print

    print()
    main()
