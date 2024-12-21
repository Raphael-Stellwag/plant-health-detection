# This is a sample Python script.
import os


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def download_data():
    import kaggle

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files('abdallahalidev/plantvillage-dataset', path='data', unzip=True)

    # Download latest version
    # path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset", path='data', unzip=True)

    # print("Path to dataset files:", path)

    # Extract data
    # kagglehub.extract_data(path)

    # print("Files extracted")

    # Create data directory if it doesn't exist
    # data_dir = 'data'
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir)

    # Load data
    # data = kagglehub.load_data()

    # print("Data loaded")

    # Save data to the data directory
    # kagglehub.save_data(data, data_dir)

    # print(f"Data saved to {data_dir}")

    # return data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    if os.path.isdir('data') and os.path.isdir('data/plantvillage dataset'):
        print("Data directory already exists, skip downloading data")
    else:
        download_data()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
