import os
import logging
import pickle
import argparse

import pickle5
import numpy as np
import pandas as pd

from tqdm import tqdm


logging.basicConfig(format='%(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(description='')
parser.add_argument('dataset_path', help='Path to generated dataset')
parser.add_argument('--n_citing_sentence', type=int, default=3, help='Number of citing sentence in validation set')
args = parser.parse_args()


def make_summarization_csv():
    logging.info('Making csv files in summarization format...')
    logging.info("Columns={'reference': cited paper abstracts (+ citing sentences), 'target': literature review chapter texts}")
    section_df = pd.read_pickle(os.path.join(args.dataset_path, 'section_survey_df.pkl'))

    introduction_ids = []
    conclusion_ids = []

    prev_id = ""
    for i, _id in enumerate(section_df["paper_id"]):
        if _id != prev_id:
            prev_id = _id
            introduction_ids.append(section_df.index[i])
            if i != 0:
                conclusion_ids.append(section_df.index[i-1])

    drop_ids = np.unique(np.concatenate([introduction_ids, conclusion_ids]))
    dataset_df = section_df.drop(index=list(drop_ids))  # Remove Introduction and Conclusion sections
    dataset_df = dataset_df[dataset_df['n_bibs'].apply(lambda n_bibs: n_bibs > 1)]  # Remove sections with less than two citations

    dataset_df = dataset_df.rename(columns={'text': 'target'})
    dataset_df = dataset_df.rename(columns={'bib_cinting_sentences': 'bib_citing_sentences'})

    if args.n_citing_sentence <= 0:
        dataset_df['reference'] = dataset_df['bib_abstracts'].apply(lambda bib_abstracts: ' '.join(['[CLS] {}. [SEP] BIB{}'.format(abstract.strip()[:-6], bib) for bib, abstract in bib_abstracts.items()]))
    else:
        dataset_df['reference'] = dataset_df[['bib_abstracts', 'bib_citing_sentences']].apply(lambda bib_abstracts: ' '.join(['[CLS] {}. [SEP] {} [SEP] BIB{}'.format(abstract.strip()[:-6], ' '.join(citing_sentence[:args.n_citing_sentence]), bib) for bib, abstract, citing_sentence in zip(bib_abstracts[0].keys(), bib_abstracts[0].values(), bib_abstracts[1].values())]), axis=1)
        #dataset_df['reference'] = dataset_df[['bib_abstracts', 'bib_citing_sentences']].apply(lambda bib_abstracts: ' '.join(['{}. BIB{}'  for abstract, citing_sentence in zip(bib_abstracts[0].items(), bib_abstracts[1].items())]))

    split_df = dataset_df['split']
    dataset_df = dataset_df[['reference', 'target']]

    train_df = dataset_df[split_df == 'train']
    val_df = dataset_df[split_df == 'val']
    test_df = dataset_df[split_df == 'test']

    train_df.to_csv(os.path.join(args.dataset_path, 'train.csv'))
    val_df.to_csv(os.path.join(args.dataset_path, 'val.csv'))
    test_df.to_csv(os.path.join(args.dataset_path, 'test.csv'))
    logging.info('Done!')


if __name__ == "__main__":
    make_summarization_csv()