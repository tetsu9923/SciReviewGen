# SciReviewGen
**This is the official dataset repository for `SciReviewGen: A Large-scale Dataset for Automatic Literature Review Generation` in ACL 2023.**

## Environment

- Python 3.9
- Run the following script to install required packages.
```
pip install -r requirements.txt
```

## 1. Preprocessing
- Download [S2ORC](https://github.com/allenai/s2orc) (We use the version released on **2020-07-05**, which contains papers up until 2020-04-14)
- Run the following command:
```
git clone https://github.com/tetsu9923/SciReviewGen.git
cd SciReviewGen
python json_to_df.py \
  -s2orc_path <Path to the S2ORC full dataset directory (Typically ".../s2orc/full/20200705v1/full")> \
  -dataset_path <Path to the generated dataset> \
  --field <the field of the literature reviews (default: "Computer Science")>
```
- The metadata and pdf parses of the literature reviews and the cited papers are stored in *dataset_path* (in the form of pandas dataframe)

## 2. Construct SciReviewGen (split)
