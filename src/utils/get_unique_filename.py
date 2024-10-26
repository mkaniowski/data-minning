import os

def get_unique_filename(base_path):
    if not os.path.exists(base_path):
        return base_path
    else:
        i = 1
        base_name, extension = os.path.splitext(base_path)
        while os.path.exists(f"{base_name}_{i}{extension}"):
            i += 1
        return f"{base_name}_{i}{extension}"