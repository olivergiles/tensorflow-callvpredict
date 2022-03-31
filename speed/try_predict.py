from tensorflow.keras.models import load_model
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import os
import time

def test_model():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_path, "models", "model.h5")
    model = load_model(model_path)
    return model

def load_data():
    X, y = make_moons(n_samples=5_000_000, noise=0.4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.3, random_state=42)
    return X_test, y_test

def main():
    model = test_model()
    X, y = load_data()
    RUNS = 5
    SIZE = 50
    times = []
    for i in range(5):
        start = time.time()
        for i in range(SIZE):
            model.predict(X[i].reshape(-1,2))
        end = time.time()
        times.append(end-start)
    print(f"""using predict a loop of {SIZE} predictions over {RUNS} runs
          took {sum(times)}""")

if __name__ == "__main__":
    main()
