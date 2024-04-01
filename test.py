from experiment_name.preprocess.base import Preprocess
from experiment_name.utils.common import import_lib
from experiment_name.utils.config import load_config

if __name__ == "__main__":
    config = load_config("config/default.yml")
    preprocess = Preprocess(
        config = config
    )
    preprocess.run()
