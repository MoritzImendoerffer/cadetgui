from pathlib import Path

def get_storage_path(base_path="~"):
    h5_dir = Path(base_path).joinpath("cadet_database").expanduser()
    h5_dir.mkdir(parents=True, exist_ok=True)
    return h5_dir