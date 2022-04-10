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
    BATCH = 5
    SIZE = 50
    call_times = []
    predict_times = []
    for _ in range(5):
        start = time.time()
        for i in range(0, SIZE, BATCH):
            model.__call__(X[i+BATCH].reshape(-1,2))
        end = time.time()
        call_times.append(end-start)
    for _ in range(5):
        start = time.time()
        for i in range(0, SIZE, BATCH):
            model.predict(X[i+BATCH].reshape(-1,2))
        end = time.time()
        predict_times.append(end-start)

    print(f"""using predict a loop of {SIZE} predictions in batches of {BATCH} over {RUNS} runs
          took {sum(call_times)}""")
    print(f"""using predict a loop of {SIZE} predictions in batches of {BATCH} over {RUNS} runs
          took {sum(predict_times)}""")

if __name__ == "__main__":
    main()
