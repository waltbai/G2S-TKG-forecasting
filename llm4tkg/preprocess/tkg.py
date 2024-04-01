import os
from datetime import date, timedelta
from typing import List, Dict

from pydantic import BaseModel

from llm4tkg.utils.config import load_config


class Quadruple(BaseModel):
    head: str
    rel: str
    tail: str
    time: str

    def __str__(self):
        return (f"({self.head},"
                f" {self.rel},"
                f" {self.tail},"
                f" {self.time})")


def read_index_file(fp: str) -> List[List[int]]:
    """Read index file."""
    result = []
    with open(fp, 'r', encoding="utf-8") as f:
        for line in f:
            result.append(list(map(int, line.split("\t"))))
    return result


def read_dict_file(
        fp: str,
        recover_space: bool = True,
) -> Dict[int, str]:
    """Read dictionary file."""
    result = {}
    with open(fp, 'r', encoding="utf-8") as f:
        for line in f:
            word, index = line.strip().split("\t")
            if recover_space:
                word = word.replace("_", " ")
            result.setdefault(int(index), word)
    return result


def idx2quadruples(
        indices: List[List[int]],
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
        base_time: str,
        time_unit: str = "day"
) -> List[Quadruple]:
    """Convert indices to quadruple."""
    result = []
    for head_idx, rel_idx, tail_idx, time_idx in indices:
        head = id2entity[head_idx]
        rel = id2relation[rel_idx]
        tail = id2entity[tail_idx]
        if time_unit == "day":
            time = str(date.fromisoformat(base_time) + timedelta(days=time_idx - 1))
        elif time_unit == "year":
            time = str(int(base_time) + time_idx - 1)
        else:
            time = "unknown"
        quad = Quadruple(
            head=head,
            rel=rel,
            tail=tail,
            time=time,
        )
        result.append(quad)
    return result


class TemporalKG(BaseModel):
    # Basic graph elements
    entities: List[str]
    relations: List[str]
    # Datasets
    train_set: List[Quadruple]
    valid_set: List[Quadruple]
    test_set: List[Quadruple]

    @classmethod
    def load(cls,
             dataset_name: str,
             data_dir: str=None,
    ):
        """Construct a temporal KG dataset."""
        # Load basic config
        config = load_config("config/default.yml")
        if data_dir is None:
            data_dir = config["data_dir"]
        # Set paths
        dataset_dir = os.path.join(data_dir, dataset_name)
        train_path = os.path.join(dataset_dir, "train.txt")
        valid_path = os.path.join(dataset_dir, "valid.txt")
        test_path = os.path.join(dataset_dir, "test.txt")
        entity2id_path = os.path.join(dataset_dir, "entity2id.txt")
        relation2id_path = os.path.join(dataset_dir, "relation2id.txt")
        # Load original files
        # In previous works, train/valid/test files are processed index files.
        train_set_idx = read_index_file(train_path)
        valid_set_idx = read_index_file(valid_path)
        test_set_idx = read_index_file(test_path)
        id2entity = read_dict_file(entity2id_path)
        id2relation = read_dict_file(relation2id_path)
        entities = [_ for _ in id2entity.values()]
        relations = [_ for _ in id2relation.values()]
        # Convert indices to quadruples
        base_time = str(config["datasets"][dataset_name]["base_time"])
        time_unit = config["datasets"][dataset_name]["time_unit"]
        train_set = idx2quadruples(
            indices=train_set_idx,
            id2entity=id2entity,
            id2relation=id2relation,
            base_time=base_time,
            time_unit=time_unit,
        )
        valid_set = idx2quadruples(
            indices=valid_set_idx,
            id2entity=id2entity,
            id2relation=id2relation,
            base_time=base_time,
            time_unit=time_unit,
        )
        test_set = idx2quadruples(
            indices=test_set_idx,
            id2entity=id2entity,
            id2relation=id2relation,
            base_time=base_time,
            time_unit=time_unit,
        )
        # Statictics
        print(f"Totally {len(entities)} entities, {len(relations)} relations.")
        print(f"Train set size: {len(train_set)}\n"
              f"Valid set size: {len(valid_set)}\n"
              f"Test set size: {len(test_set)}")
        print(f"Some examples:")
        for item in train_set[:10]:
            print(item)
        return cls(
            entities=entities,
            relations=relations,
            train_set=train_set,
            valid_set=valid_set,
            test_set=test_set,
        )

