# SciReviewGen
**This is the official dataset repository for [SciReviewGen: A Large-scale Dataset for Automatic Literature Review Generation](https://arxiv.org/pdf/2305.15186.pdf) in ACL findings 2023.**

## Dataset
- [split_survey_df](https://drive.google.com/file/d/1S6v-xaCDND4ilK38sEpkfcOoMnffX7Zf/view?usp=sharing): The split version of SciReviewGen, which aims to generate literature review **chapters**
- [original_survey_df](https://drive.google.com/file/d/1MnjQ2fQ_fJjcqKvIwj2w7P6IGh4GszXH/view?usp=sharing): The original version of SciReviewGen, which aims to generate **the entire text** of literature reviews
- [summarization_csv](https://drive.google.com/file/d/1okvILkxfrpTQYWLxbV4lM9BQnuVaAfbY/view?usp=sharing): CSV files suitable for summarization task. You can apply them to [HuggingFace's official sample codes](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization#custom-csv-files)

### Data format
#### split_survey_df & original_survey_df
- Row: 
  - literature review chapter or the entire text of literature review
- Column:
  - paper_id: paper_id used in [S2ORC](https://github.com/allenai/s2orc)
  - title: title of the literature review
  - abstract: abstract of the literature review
  - section: chapter title
  - text: body text of literature review chapter or literature review paper
  - n_bibs: number of the cited papers that can be used as inputs
  - n_nonbibs: number of the cited papers that cannot be used as inputs
  - bib_titles: titles of the cited papers
  - bib_abstracts: abstracts of the cited papers
  - bib_citing_sentences: citing sentences that cite the cited papers
  - split: train/val/test split

#### summarization_csv
- Row: 
  - literature review chapter
- Column:
  - reference: `literature review title <s> chapter title <s> abstract of cited paper 1 <s> BIB001 </s> literature review title <s> chapter title <s> abstract of cited paper 2 <s> BIB002 </s> ...`
  - target: literature review chapter


## How to create SciReviewGen from S2ORC
### 0. Environment
- Python 3.9
- Run the following command to clone the repository and install the required packages
```
git clone https://github.com/tetsu9923/SciReviewGen.git
cd SciReviewGen
pip install -r requirements.txt
```

### 1. Preprocessing
- Download [S2ORC](https://github.com/allenai/s2orc) (We use the version released on **2020-07-05**, which contains papers up until 2020-04-14)
- Run the following command:
```
python json_to_df.py \
  -s2orc_path <Path to the S2ORC full dataset directory (Typically ".../s2orc/full/20200705v1/full")> \
  -dataset_path <Path to the generated dataset> \
  --field <Optional: the field of the literature reviews (mag_field_of_study in S2ORC, default="Computer Science")>
```
The metadata and pdf parses of the candidates for the literature reviews and the cited papers are stored in *dataset_path* (in the form of pandas dataframe).

### 2. Construct SciReviewGen
- Run the following command:
```
python make_section_df.py \
  -dataset_path <Path to the generated dataset> \
  --version <Optional: the version of SciReviewGen ("split" or "original", default="split")>
```
The SciReviewGen dataset (**split_survey_df.pkl** or **original_survey_df.pkl**) is stored in *dataset_path* (in the form of pandas dataframe).
`filtered_dict.pkl` gives the list of literature reviews after filtering by the [SciBERT](https://arxiv.org/abs/1903.10676)-based classifier (Section 3.2).

### 3. Construct csv data for summarization
- Run the following command:
```
python make_summarization_csv.py \
  -dataset_path <Path to the generated dataset> 
```
The csv files for summarization (**train.csv**, **val.csv**, and **test.csv**) are stored in *dataset_path*.
If you train QFiD on the generated csv files, add `--for_qfid` argument as below.
```
python make_summarization_csv.py \
  -dataset_path <Path to the generated dataset> \
  --for_qfid
```


## Additional resources
### SciBERT-based literature review classifier
We trained the [SciBERT](https://arxiv.org/abs/1903.10676)-based literature review classifier.
The model weights are available [here](https://drive.google.com/file/d/1cPGJpvCFQkHX2td99YyFitBirG-eCcLC/view?usp=sharing).

### Query-weighted Fusion-in-Decoder (QFiD)
We proposed Query-weighted Fusion-in-Decoder (QFiD) that explicitly considers the relevance of each input document to the queries.
You can train QFiD on SciReviewGen csv data (**Make sure that you passed** `--for_qfid` **argument when executing** `make_summarization_csv.py`).
#### Train
- Modify qfid/train.sh (CUDA_VISIBLE_DEVICES, csv file path, outpput_dir, and num_train_epochs)
- Run the following command:
```
cd qfid
./train.sh
```
#### Test
- Modify qfid/test.sh (CUDA_VISIBLE_DEVICES, csv file path, outpput_dir, and num_train_epochs. **Please set *num_train_epochs* as the number of epochs you trained in total**)
- Run the following command:
```
./test.sh
```

## Licenses
- SciReviewGen is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). **You can use SciReviewGen for only non-commercial purposes.**
- SciReviewGen is created based on [S2ORC](https://github.com/allenai/s2orc). Note that S2ORC is released under CC BY-NC 4.0, which allows users to copy and redistribute for only non-commercial purposes.
