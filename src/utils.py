import os

def create_directory(dir_name):
    """Creates a directory if it does not exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Create necessary directories
create_directory("visualizations")
