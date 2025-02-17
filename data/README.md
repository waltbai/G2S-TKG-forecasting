# Introduction
Datasets are processed from [RE-GCN](https://github.com/Lee-zix/RE-GCN).

## Changes
We made following changes:

### ICEWS14
- Underline ("_") are replaced by space (" "), so that entities and relations are more like natural language.
- Time index are deducted by 1, so that timestamps start from 0.

### ICEWS18
- Underline ("_") are replaced by space (" "), so that entities and relations are more like natural language.
- Time index are diveded by 24, so that the time unit is DAY now.
- The 5-th column is removed.

### ICEWS05-15
- Underline ("_") are replaced by space (" "), so that entities and relations are more like natural language.

### GDELT
- Brackets ("()") in entities are removed.
- Tags in entities are removed ("@***" in brackets).
- Entities are converted to Title-format.
- Relation are mapped from CAMEO event code to actual relation name.
- Time index are divided by 15, so that the time unit is 15 minutes now.

### YAGO
- Brackets ("<>") are removed.
- Underline ("_") are replaced by space (" "), so that entities and relations are more like natural language.

## Process command
If you want to process the dataset from original RE-GCN format:
```shell
python dataset_converter.py \
    --input_dir /path/to/specific/dataset \
    --output_dir .
```

For example:
```shell
python dataset_converter.py \
    --input_dir ../../RE-GCN/data/ICEWS14s \
    --output_dir . \
    --dataset ICEWS14 \
    --cameo_path cameo.yaml
```

The argument are listed as follows:

| Argument   | Type                            | Value                                                                         |
|------------|---------------------------------|-------------------------------------------------------------------------------|
| input_dir  | str, required                   | path to a specific RE-GCN dataset                                             |
| output_dir | str, default to be "."          | path to output dataset directory, a sub-folder will automatically be created. |
| dataset    | str, default to be None         | specify the dataset name, if None then program will guess the dataset name    |
| cameo_path | str, default to be "cameo.yaml" | path to a cameo event code file.                                              |
