from tensorflow.keras.models import load_model
import pandas as pd
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

def batch_run(model, batch):
    X, y = load_data()
    RUNS = 5
    BATCH = batch
    SIZE = 1_000
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

    print(f"""using call a loop of {SIZE} predictions in batches of {BATCH} over {RUNS} runs
          took {sum(call_times)}""")
    print(f"""using predict a loop of {SIZE} predictions in batches of {BATCH} over {RUNS} runs
          took {sum(predict_times)}""")
    return sum(call_times), sum(predict_times)

def main():
    model = test_model()
    to_test = [1, 5, 10, 20, 50, 100, 250, 500, 1_000]
    results = {
        "size":[],
        "call":[],
        "predict":[]
    }
    for batch in to_test:
        results["size"].append(batch)
        call, predict = batch_run(model, batch)
        results["call"].append(call)
        results["predict"].append(predict)
    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()
