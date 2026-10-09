from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main():
    # Wczytanie danych
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # Podział na zbiór treningowy i testowy (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Inicjalizacja i trening modelu
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model Regresji Logistycznej został pomyślnie wytrenowany na zbiorze treningowym.")

if __name__ == "__main__":
    main()