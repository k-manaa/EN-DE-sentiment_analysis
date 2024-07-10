import os
from sentiment_analysis.cleaner import clean_folder


def test_clean_folder(tmpdir):
    # Create a temporary folder for testing
    folder_path = str(tmpdir.mkdir("test_folder"))

    # Create test files
    file_names = ['ENreviews.csv', 'DEreviews.csv', 'file1.txt', 'file2.txt']
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        open(file_path, 'w').close()

    # Call the clean_folder function
    clean_folder(folder_path)

    # Check if the unwanted files are deleted and the desired files are preserved
    assert not os.path.exists(os.path.join(folder_path, 'file1.txt'))
    assert not os.path.exists(os.path.join(folder_path, 'file2.txt'))
    assert os.path.exists(os.path.join(folder_path, 'ENreviews.csv'))
    assert os.path.exists(os.path.join(folder_path, 'DEreviews.csv'))
