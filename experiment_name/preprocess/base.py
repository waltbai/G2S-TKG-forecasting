import os
from abc import ABC, abstractmethod
from typing import Dict, List

from experiment_name.utils.common import import_lib


def create_path(
        config: Dict,
        key: str,
        default: str | List[str] = None
) -> str:
    """Find key in config, otherwise use default path."""
    if key in config:
        return config[key]
    elif default is not None:
        return default
    else:
        raise ValueError(f"Please provide `{key}` in config.")


class Preprocess:
    """Preprocess class."""

    def __init__(
            self,
            config: Dict,
    ):
        self._config = config
        # Set source data path
        self.data_path = config["data_path"]
        # Set workspace path
        self.workspace_path = create_path(config, "workspace_path", "preprocess")
        if not os.path.exists(self.workspace_path):
            os.makedirs(self.workspace_path)
        # Set train/dev/test paths
        self.train_path = create_path(config, "train_path", "data/train.json")
        self.dev_path = create_path(config, "dev_path", "data/dev.json")
        self.test_path = create_path(config, "test_path", "data/test.json")
        # Construct preprocess pipeline
        self._pipeline = []
        for num, component_config in enumerate(config["preprocess"]["components"]):
            path = "experiment_name.preprocess." + component_config["path"]
            component_cls = import_lib(path)
            # Set input path
            if "depends" not in component_config or \
                    not component_config["depends"]:
                input_path = self.data_path
            else:
                input_path = [
                    os.path.join(self.workspace_path, _)
                    for _ in component_config["depends"]
                ]
            # Set output path
            if num != len(config["preprocess"]["components"]) - 1:
                output_path = os.path.join(
                    self.workspace_path, component_config["name"])
                component = component_cls(
                    name=component_config["name"],
                    input_path=input_path,
                    workspace_path=self.workspace_path,
                    output_path=output_path
                )
            else:
                component = component_cls(
                    name=component_config["name"],
                    input_path=input_path,
                    workspace_path=self.workspace_path,
                    train_path=self.train_path,
                    dev_path=self.dev_path,
                    test_path=self.test_path,
                )
            self._pipeline.append(component)

    def run(self, **kwargs):
        """Run preprocess pipeline."""
        for num, step in enumerate(self._pipeline):
            # Define input_path and output_path
            # Run component
            if num != len(self._pipeline) - 1:
                step.run(
                    input_path = step.input_path,
                    output_path = step.output_path,
                )
            else:
                step.run(
                    input_path = step.input_path,
                    train_path = step.train_path,
                    dev_path = step.dev_path,
                    test_path = step.test_path,
                )


class AbstractPreprocessComponent(ABC):
    """Abstract preprocess component class."""

    def __init__(
            self,
            name: str = "",
            input_path: str | List[str] = None,
            output_path: str = None,
            workspace_path: str = None,
            **kwargs,
    ):
        self._name = name
        # Set workspace path
        self._workspace_path = workspace_path
        # Set input path
        self.input_path = input_path
        # Set output path
        self.output_path = output_path

    @abstractmethod
    def run(self,
            input_path: str | List[str],
            output_path: str,
            **kwargs):
        """Main process function."""


class AbstractPreprocessCollector(AbstractPreprocessComponent, ABC):
    """Abstract preprocess collector class."""

    def __init__(
            self,
            train_path: str = None,
            dev_path: str = None,
            test_path: str = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

    @abstractmethod
    def run(self,
            input_path: str | List[str],
            train_path: str = None,
            dev_path: str = None,
            test_path: str = None,
            **kwargs):
        """Main process function."""
