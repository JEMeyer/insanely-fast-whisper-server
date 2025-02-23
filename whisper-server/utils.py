import os


def save_temp_file(contents, file_name):
    with open(file_name, "wb") as temp_file:
        temp_file.write(contents)
    return file_name


def remove_temp_file(file_name):
    os.remove(file_name)
