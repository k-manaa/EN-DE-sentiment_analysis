import os
import argparse

def clean_folder(folder_paths: str) -> None:
    """
    Deletes all files in the specified folders that are not named "ENreviews.csv" or "DEreviews.csv".
    """
    for folder_path in folder_paths:
        # Get the list of files in the folder
        file_list = os.listdir(folder_path)

        # Iterate over the files and delete those that do not match the desired names
        for file_name in file_list:
            if file_name != 'ENreviews.csv' and file_name != 'DEreviews.csv':
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description='Clean folders by deleting unwanted files.')
    parser.add_argument('folder_paths', nargs='+', type=str, help='Paths to the folders')
    args = parser.parse_args()

    # Call the function to clean the folders
    clean_folder(args.folder_paths)

if __name__ == "__main__":
    main()
