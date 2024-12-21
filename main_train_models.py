# This is a sample Python script.
import os

from model_template import ModelTemplate


def download_data():
    import kaggle

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files('abdallahalidev/plantvillage-dataset', path='data', unzip=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if os.path.isdir('data') and os.path.isdir('data/plantvillage dataset'):
        print("Data directory already exists, skip downloading data")
    else:
        if not os.path.isdir('data'):
            os.mkdir('data')
        download_data()

    if not os.path.isdir('models'):
        os.mkdir('models')

    if not os.path.isdir('output'):
        os.mkdir('output')

    models = [ModelTemplate()]

    for model in models:
        model.train()
        model.save_model()