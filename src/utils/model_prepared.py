from train import entrenar_modelo
import pandas as pd

def preparar_modelo():
    train = pd.read_csv("../data/processed/train.csv")
    test = pd.read_csv("../data/processed/test.csv")

    train = train.drop('Unnamed: 0', axis=1)
    test = test.drop('Unnamed: 0', axis=1)
    entrenar_modelo(train, test)
