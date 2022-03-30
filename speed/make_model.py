from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
import os

def create_model():
    """Basic binary classification model with the sequential API"""
    inputs = Input(shape=(2,))
    x = Dense(32, activation="relu", input_dim=2)(inputs)
    for _ in range(50):
        x = Dense(32)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def main():
    model = create_model()

    EPOCHS = 50
    X, y = make_moons(n_samples=5000, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.3, random_state=42)
    model = create_model()
    model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_split=0.3, batch_size=32)
    model.evaluate(X_test, y_test)

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_path, "models", "model.h5")
    model.save(model_path)

if __name__ == "__main__":
    main()