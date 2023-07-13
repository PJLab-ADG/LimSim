import yaml


def load_config(config_file_path: str) -> dict:
    """Loads the configuration file from the given path.
    
    Args:
        config_file_path (str): Path to the configuration file.
    
    Returns:
        dict: The configuration dictionary.
    """
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
