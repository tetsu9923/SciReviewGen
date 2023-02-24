import os
import logging
import pickle
import argparse

import pandas as pd

from tqdm import tqdm


def save_metadata_survey_df(args):
    logging.info('Loading metadata of literature reviews...')
    keywords = ['survey', 'overview', 'literature review', 'a review']

    metadata_survey_dfs = []
    for i in tqdm(range(100)):
        metadata_df = pd.read_json(os.path.join(args.s2orc_path, 'metadata/metadata_{}.jsonl.gz'.format(i)), lines=True, compression='infer')
        metadata_survey_df = metadata_df[
                        (metadata_df.mag_field_of_study.apply(lambda field: args.field in field if field is not None else False))
                        & (metadata_df.title.apply(lambda title: any([word in title.lower() for word in keywords])))
                        & metadata_df.has_outbound_citations
                        & metadata_df.has_pdf_body_text
                        & ~metadata_df.abstract.isna()
        ]
        metadata_survey_dfs.append(metadata_survey_df)
    metadata_survey_df = pd.concat(metadata_survey_dfs)
    metadata_survey_df = metadata_survey_df.set_index('paper_id')
    metadata_survey_df.index = metadata_survey_df.index.astype('str')
    metadata_survey_df.to_pickle(os.path.join(args.dataset_path, 'metadata_survey_df.pkl'))
    logging.info('Done!')

def save_metadata_outbound_df(args):
    logging.info('Loading metadata of cited papers...')
    metadata_survey_df = pd.read_pickle(os.path.join(args.dataset_path, 'metadata_survey_df.pkl'))
    outbound_paper_ids = set([paper_id for paper_ids in metadata_survey_df.outbound_citations.values for paper_id in paper_ids])

    metadata_outbound_dfs = []
    for i in tqdm(range(100)):
        metadata_df = pd.read_json(os.path.join(args.s2orc_path, 'metadata/metadata_{}.jsonl.gz'.format(i)), lines=True, compression='infer')
        metadata_df = metadata_df.set_index('paper_id')
        metadata_df.index = metadata_df.index.astype('str')
        cs_survey_outbound_paper_ids = list(outbound_paper_ids & set(metadata_df.index))
        metadata_outbound_df = metadata_df.loc[cs_survey_outbound_paper_ids]
        metadata_outbound_dfs.append(metadata_outbound_df)
    metadata_outbound_df = pd.concat(metadata_outbound_dfs)
    metadata_outbound_df.to_pickle(os.path.join(args.dataset_path, 'metadata_outbound_df.pkl'))
    logging.info('Done!')

def save_pdf_df(args):
    logging.info('Loading pdf parses of literature review papers and cited papers...')
    metadata_survey_df = pd.read_pickle(os.path.join(args.dataset_path, 'metadata_survey_df.pkl'))
    survey_index = metadata_survey_df.index
    outbound_index = pd.read_pickle(os.path.join(args.dataset_path, 'metadata_outbound_df.pkl')).index
    
    pdf_survey_dfs = []
    pdf_outbound_dfs = []
    for i in tqdm(range(100)):
        pdf_df = pd.read_json(os.path.join(args.s2orc_path, 'pdf_parses/pdf_parses_{}.jsonl.gz'.format(i)), lines=True, compression='infer')
        pdf_df = pdf_df.set_index('paper_id')
        pdf_df.index = pdf_df.index.astype('str')
    
        pdf_survey_paper_ids = list(set(survey_index) & set(pdf_df.index))
        pdf_survey_df = pdf_df.loc[pdf_survey_paper_ids]
        pdf_survey_df = pdf_survey_df[['body_text', 'bib_entries']]

        pdf_survey_df['title'] = ''
        pdf_survey_df['abstract'] = ''
        for i, row in enumerate(pdf_survey_df.itertuples()):
            pdf_survey_df.at[pdf_survey_df.index[i], 'title'] = metadata_survey_df.query('paper_id == @row.Index')['title'].item()
            pdf_survey_df.at[pdf_survey_df.index[i], 'abstract'] = metadata_survey_df.query('paper_id == @row.Index')['abstract'].item()
        pdf_survey_dfs.append(pdf_survey_df)
    
        pdf_outbound_paper_ids = list(set(outbound_index) & set(pdf_df.index))
        pdf_outbound_df = pdf_df.loc[pdf_outbound_paper_ids]
        pdf_outbound_df = pdf_outbound_df[pdf_outbound_df.body_text.apply(lambda text: len(text) > 0)]
        pdf_outbound_df = pdf_outbound_df[['body_text', 'bib_entries']]
        pdf_outbound_dfs.append(pdf_outbound_df)
    
    pdf_survey_df = pd.concat(pdf_survey_dfs)
    pdf_survey_df.to_pickle(os.path.join(args.dataset_path, 'pdf_survey_df.pkl'))
    pdf_outbound_df = pd.concat(pdf_outbound_dfs)
    pdf_outbound_df.to_pickle(os.path.join(args.dataset_path, 'pdf_outbound_df.pkl'))
    logging.info('Done!')

    
if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s2orc_path', help='Path to the S2ORC full dataset directory (Typically ".../s2orc/full/20200705v1/full")')
    parser.add_argument('-dataset_path', help='Path to the generated dataset')
    parser.add_argument('--field', default='Computer Science', help='Field of literature reviews')
    args = parser.parse_args()

    save_metadata_survey_df(args)  # collect metadata of the literature reviews
    save_metadata_outbound_df(args)  # collect metadata of the cited papers
    save_pdf_df(args)  # collect pdf parses of the literature reviews and the cited papers