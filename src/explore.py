from sklearn.datasets import load_iris
import pandas as pd

def main():
    # Wczytanie zbioru danych Iris jako DataFrame
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Wyświetlenie pierwszych wierszy
    print("Pierwsze 5 wierszy:")
    print(df.head(), "\n")

    # Wyświetlenie rozmiaru zbioru
    print("Rozmiar zbioru danych:")
    print(f"Liczba wierszy i kolumn: {df.shape}\n")

    # Wyświetlenie podstawowych statystyk opisowych
    print("Statystyki opisowe:")
    print(df.describe())

if __name__ == "__main__":
    main()