from pathlib import Path
import datetime

def get_storage_path(base_path="~"):
    h5_dir = Path(base_path).joinpath("cadet_database").expanduser()
    h5_dir.mkdir(parents=True, exist_ok=True)
    return h5_dir

def get_experiment_path():
    now = datetime.datetime.now().strftime("%Y-%h-%d-%H-%M-%S")
    h5_dir = get_storage_path()
    h5_dir.joinpath(now)
    h5_dir.mkdir(parents=True, exist_ok=True)
    return h5_dir