from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Wczytanie danych i podział
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Inicjalizacja i trening modelu
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = model.predict(X_test)
    
    # Wyznaczenie metryk
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=iris.target_names)

    # Wyświetlenie w terminalu
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(report)

    # Zapis wyników do pliku wyniki.txt
    with open("wyniki.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print("Wyniki zostały pomyślnie zapisane do pliku wyniki.txt.")

if __name__ == "__main__":
    main()