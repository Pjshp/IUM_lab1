from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    # Wczytanie danych
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Rozmiar zbioru treningowego: {X_train.shape[0]} wierszy")
    print(f"Rozmiar zbioru testowego: {X_test.shape[0]} wierszy\n")

    # Inicjalizacja i trening modelu
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model Regresji Logistycznej został pomyślnie wytrenowany.")

    # Sprawdzenie podstawowej dokładności na zbiorze testowym
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność modelu (accuracy) na zbiorze testowym: {accuracy:.2f}")

if __name__ == "__main__":
    main()