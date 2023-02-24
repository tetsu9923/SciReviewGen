import os
import logging
import pickle
import argparse

import nltk.data
import numpy as np
import pandas as pd

from tqdm import tqdm


def append_citing_sentence(args):
    extra_abbreviations = ['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'i.e', 'al']
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer._params.abbrev_types.update(extra_abbreviations)

    logging.info('Loading citing sentences...')
    cs_survey_ids = pd.read_pickle(os.path.join(args.dataset_path, 'metadata_survey_df.pkl')).index
    pdf_outbound_df = pd.read_pickle(os.path.join(args.dataset_path, 'pdf_outbound_df.pkl'))
    metadata_outbound_df = pd.read_pickle(os.path.join(args.dataset_path, 'metadata_outbound_df.pkl'))
    metadata_outbound_df = metadata_outbound_df.dropna(subset=['abstract'])
    
    citing_df = pdf_outbound_df.copy()
    citing_df['bib_entries'] = citing_df['bib_entries'].apply(
        lambda bib_entries: {key: value['link'] for key, value in bib_entries.items()}
    )
    citing_df['bib_mark'] = citing_df['body_text'].apply(
        lambda body_text: [cite['text'] for cite in sum([body['cite_spans'] for body in body_text], [])]
    )
    
    citing_dict = {}
    bar = tqdm(total=len(citing_df))
    # citing_dict: {paper A: {paper B cited in A: 'citing sentence in A (explaining B)', paper C cited in A: 'citing sentence in A (explaining C)'}, paperD: ...}
    for i, row in enumerate(citing_df.itertuples()):
        tmp_dict = {}
        for section in row.body_text:
            section_text = section['text'].split()
            for citing in section['cite_spans']:
                for sentence in tokenizer.tokenize(section['text']):
                    if citing['text'] in sentence:
                        text = sentence
                        if citing['ref_id'] in row.bib_entries.keys():
                            if row.bib_entries[citing['ref_id']] != None:
                                tmp_dict[row.bib_entries[citing['ref_id']]] = text
                        break
        citing_dict[citing_df.index[i]] = tmp_dict
        bar.update(1)
    logging.info('Done!')

    logging.info('Appending citing sentences to metadata_outbound_df...')    
    citing_sentence_list = []
    bar = tqdm(total=len(metadata_outbound_df))
    for index, citing_papers in metadata_outbound_df['inbound_citations'].iteritems():
        citing_sentence = []
        n_of_citing = 0
        for citing_paper in citing_papers:
            if citing_paper in citing_dict.keys() and citing_paper not in cs_survey_ids:
                if index in citing_dict[citing_paper].keys():
                    citing_sentence.append(citing_dict[citing_paper][index])
                    n_of_citing += 1
        citing_sentence_list.append(citing_sentence)
        bar.update(1)
        
    metadata_outbound_df['citing_sentence'] = ''
    bar = tqdm(total=len(metadata_outbound_df))
    for i, row in enumerate(metadata_outbound_df.itertuples()):
        append_sentence = citing_sentence_list[i]
        metadata_outbound_df.at[metadata_outbound_df.index[i], 'citing_sentence'] = append_sentence
        bar.update(1)
        
    metadata_outbound_df.to_pickle(os.path.join(args.dataset_path, 'metadata_outbound_citation_df.pkl'))
    logging.info('Done!')


def make_scireviewgen(args):
    logging.info('Making section_df...')
    metadata_outbound_df = pd.read_pickle(os.path.join(args.dataset_path, 'metadata_outbound_citation_df.pkl'))
    metadata_outbound_df = metadata_outbound_df.dropna(subset=['abstract'])
    pdf_survey_df = pd.read_pickle(os.path.join(args.dataset_path, 'pdf_survey_df.pkl'))
    pdf_survey_df = pdf_survey_df[pdf_survey_df['abstract'].apply(lambda s: type(s) == str)]  

    def get_section_df(row):
        sections_duplicate = [paragraph['section'] for paragraph in row.body_text]
        sections = sorted(set(sections_duplicate), key=sections_duplicate.index)
        bib_df = pd.DataFrame.from_dict(row.bib_entries, orient='index')
        bib_df = bib_df[bib_df.link.apply(lambda paper_id: paper_id in metadata_outbound_df.index)]
        bib_dict = bib_df.link.dropna().to_dict()

        def replace_cite(body_row):
            body_text = ''
            start_index = 0
            for cite_span in body_row.cite_spans:
                end_index = cite_span['start']
                ref_id = cite_span['ref_id']
                body_text += body_row.text_raw[start_index:end_index]
                body_text += 'BIB{}'.format(bib_dict[ref_id]) if ref_id in bib_dict else ''
                start_index = cite_span['end']
            body_text += body_row.text_raw[start_index:]
            return body_text
    
        body_df = pd.DataFrame(row.body_text).rename(columns={'text': 'text_raw'})
        body_df['text'] = body_df[['text_raw', 'cite_spans']].apply(replace_cite, axis=1)
        body_df['title'] = row.title
        body_df['abstract'] = row.abstract
    
        section_df = body_df.groupby('section').agg({
            'text': lambda text_series: ' '.join([text for text in text_series]),
            'title': lambda text_series: [text for text in text_series][0],
            'abstract': lambda text_series: [text for text in text_series][0],
            'cite_spans': lambda cite_spans_series: [cite['ref_id'] for cite_spans in cite_spans_series for cite in cite_spans],
        })
        section_df = section_df.loc[sections]
        section_df['bibs'] = section_df['cite_spans'].apply(lambda spans: [bib_dict[span] for span in spans if span in bib_dict])
        section_df['n_bibs'] = section_df[['cite_spans', 'bibs']].apply(lambda row: len(row['bibs']), axis=1)
        section_df['n_nonbibs'] = section_df[['cite_spans', 'bibs']].apply(lambda row: len(row['cite_spans']) - len(row['bibs']), axis=1)

        section_df['paper_id'] = row.name
        section_df['section_id'] = section_df['paper_id'] + '-' + np.arange(len(section_df)).astype('str')
        section_df['section'] = section_df.index
        section_df = section_df.set_index('section_id')

        section_df['bib_titles'] = section_df['bibs'].apply(lambda bibs: metadata_outbound_df.loc[bibs]['title'].to_dict())
        section_df['bib_abstracts'] = section_df['bibs'].apply(lambda bibs: metadata_outbound_df.loc[bibs]['abstract'].to_dict())
        section_df['bib_years'] = section_df['bibs'].apply(lambda bibs: metadata_outbound_df.loc[bibs]['year'].to_dict())  
        section_df['bib_abstracts'] = section_df[['bib_abstracts', 'bib_years']].apply(lambda bib: dict(sorted(bib[0].items(), key=lambda x: bib[1][x[0]])), axis=1)  # Sort by publication year
        section_df['bib_citing_sentences'] = section_df['bibs'].apply(lambda bibs: metadata_outbound_df.loc[bibs]['citing_sentence'].to_dict())
        section_df = section_df[['paper_id', 'title', 'abstract', 'section', 'text', 'n_bibs', 'n_nonbibs', 'bib_titles', 'bib_abstracts', 'bib_citing_sentences']]
        return section_df

    section_survey_df = pd.concat(pdf_survey_df.apply(get_section_df, axis=1).values)
    section_survey_df = section_survey_df[section_survey_df['text'].apply(len) >= 1]  # Remove sections without body text

    with open ('filtered_dict.pkl', 'rb') as f:
        filtering_dict = pickle.load(f)
    section_survey_df = section_survey_df[section_survey_df['paper_id'].isin(filtering_dict.keys())]
    section_survey_df['split'] = section_survey_df['paper_id'].apply(lambda s: filtering_dict[s])

    if args.version == 'split':
        section_survey_df = section_survey_df[section_survey_df['bib_abstracts'].apply(lambda _dict: len(_dict) >= 2)]  # Remove sections with less than two cited papers
        section_survey_df.to_pickle(os.path.join(args.dataset_path, 'split_survey_df.pkl'))
    else:
        section_survey_df = section_survey_df.groupby('paper_id').agg({
            'title': lambda l: list(l)[0],
            'abstract': lambda l: list(l)[0],
            'section': lambda l: list(l),
            'text': lambda l: list(l),
            'n_bibs': lambda l: list(l),
            'n_nonbibs': lambda l: list(l),
            'bib_titles': lambda l: list(l),
            'bib_abstracts': lambda l: list(l),
            'bib_citing_sentences': lambda l: list(l),
            'split': lambda l: list(l)[0],
        })
        section_survey_df.to_pickle(os.path.join(args.dataset_path, 'original_survey_df.pkl'))

    logging.info('Done!')


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-dataset_path', help='Path to the generated dataset')
    parser.add_argument('--version', default='split', help='Specify the version ("split" or "original")',  choices=['split', 'original'])
    parser.add_argument('--n_val', type=int, default=1000, help='Number of literature review papers in validation set')
    parser.add_argument('--n_test', type=int, default=1000, help='Number of literature review papers in test set')
    args = parser.parse_args()

    append_citing_sentence(args)  # collect citing sentences for the cited papers
    make_scireviewgen(args)  # make scireviewgen dataset in the form of pandas dataframe