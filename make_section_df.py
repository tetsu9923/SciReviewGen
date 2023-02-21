# To do: 最初からアブストあるデータに絞る line.95


import os
import logging
import pickle
import argparse

import pickle5
import nltk.data
import numpy as np
import pandas as pd

from tqdm import tqdm


extra_abbreviations = ['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'i.e', 'al']
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer._params.abbrev_types.update(extra_abbreviations)

logging.basicConfig(format='%(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(description='')
parser.add_argument('dataset_path', help='Path to generated dataset')
parser.add_argument('--n_val', type=int, default=1000, help='Number of literature review papers in validation set')
parser.add_argument('--n_test', type=int, default=1000, help='Number of literature review papers in test set')
args = parser.parse_args()


def append_citing_sentence():
    logging.info('Loading citing sentences...')
    cs_survey_ids = pd.read_pickle(os.path.join(args.dataset_path, 'metadata_survey_df.pkl')).index
    pdf_outbound_df = pd.read_pickle(os.path.join(args.dataset_path, 'pdf_outbound_df.pkl'))
    metadata_outbound_df = pd.read_pickle(os.path.join(args.dataset_path, 'metadata_outbound_df.pkl'))
    metadata_outbound_df = metadata_outbound_df.dropna(subset=['abstract'])
    
    citing_df = pdf_outbound_df.copy()
    citing_df['bib_entries'] = citing_df['bib_entries'].apply(
        lambda bib_entries: {key: value["link"] for key, value in bib_entries.items()}
    )
    citing_df['bib_mark'] = citing_df['body_text'].apply(
        lambda body_text: [cite["text"] for cite in sum([body["cite_spans"] for body in body_text], [])]
    )
    
    citing_dict = {}
    bar = tqdm(total=len(citing_df))
    # {paper A: {paper B cited in A: "citing sentence in A (explaining B)", paper C cited in A: "citing sentence in A (explaining C)"}, paperD: ...}
    # {pdf_outbound_df内のpaper_id(A): {Aが引用している論文1: "1の引用文", Aが引用している論文2: "2の引用文"}, pdf_outbound_df内のpaper_id(B) ...}
    for i, row in enumerate(citing_df.itertuples()):  # 各引用について前後n文字を抽出
        tmp_dict = {}
        for section in row.body_text:  # 章ごとに
            section_text = section["text"].split()
            for citing in section["cite_spans"]:  # 引用ごとに
                for sentence in tokenizer.tokenize(section["text"]):  # 文章ごとに
                    if citing["text"] in sentence:
                        text = sentence
                        if citing["ref_id"] in row.bib_entries.keys():
                            if row.bib_entries[citing["ref_id"]] != None:
                                tmp_dict[row.bib_entries[citing["ref_id"]]] = text
                        break
        citing_dict[citing_df.index[i]] = tmp_dict
        bar.update(1)
    logging.info('Done!')

    logging.info('Appending citing sentences to metadata_outbound_df...')    
    citing_sentence_list = []  # i番目のmetadata_outbound_dfの論文（＝サーベイ論文で紹介されている論文）が引用された時の文章
    bar = tqdm(total=len(metadata_outbound_df))
    for index, citing_papers in metadata_outbound_df['inbound_citations'].iteritems():  # index: paper_id, citing_papers: 引用されている論文のpaper_idリスト
        citing_sentence = []
        n_of_citing = 0
        for citing_paper in citing_papers:
            if citing_paper in citing_dict.keys() and citing_paper not in cs_survey_ids:
                if index in citing_dict[citing_paper].keys():
                    citing_sentence.append(citing_dict[citing_paper][index])
                    n_of_citing += 1
        citing_sentence_list.append(citing_sentence)
        bar.update(1)
        
    metadata_outbound_df["citing_sentence"] = ""
    bar = tqdm(total=len(metadata_outbound_df))
    for i, row in enumerate(metadata_outbound_df.itertuples()):
        append_sentence = citing_sentence_list[i]
        metadata_outbound_df.at[metadata_outbound_df.index[i], "citing_sentence"] = append_sentence
        bar.update(1)
        
    metadata_outbound_df.to_pickle(os.path.join(args.dataset_path, 'metadata_outbound_citation_df.pkl'))
    logging.info('Done!')


def make_section_df():
    logging.info('Making section_df...')
    metadata_outbound_df = pd.read_pickle(os.path.join(args.dataset_path, 'metadata_outbound_citation_df.pkl'))
    metadata_outbound_df = metadata_outbound_df.dropna(subset=['abstract'])
    pdf_survey_df = pd.read_pickle(os.path.join(args.dataset_path, 'pdf_survey_df.pkl'))
    pdf_survey_df = pdf_survey_df[pdf_survey_df["abstract"].apply(lambda s: type(s) == str)]  # 正しく動くか確認

    test_list = [
        "Summarization from Medical Documents: A Survey",
        "Text Summarization Techniques: A Brief Survey",
        "A Survey of Text Summarization Extractive Techniques",
        "Automatic Keyword Extraction for Text Summarization: A Survey",
        "A Survey For Multi-Document Summarization",
        "A Survey of Unstructured Text Summarization Techniques",

        "A Comprehensive Survey on Graph Neural Networks",
        "Graph neural networks: A review of methods and applications",
    
        "Graph Embedding Techniques, Applications, and Performance: A Survey",
        "A Comprehensive Survey of Graph Embedding: Problems, Techniques, and Applications"
    ]

    n_val = args.n_val
    n_test = args.n_test
    assert n_test >= len(test_list)

    pdf_survey_df["split"] = ""
    n_train = len(pdf_survey_df) - (n_val + n_test)
    for i, row in enumerate(pdf_survey_df.itertuples()):
        if row.title in test_list:
            pdf_survey_df.at[pdf_survey_df.index[i], "split"] = "test"
        elif n_train > 0:
            pdf_survey_df.at[pdf_survey_df.index[i], "split"] = "train"
            n_train -= 1
        elif n_val > 0:
            pdf_survey_df.at[pdf_survey_df.index[i], "split"] = "val"
            n_val -= 1
        else:
            pdf_survey_df.at[pdf_survey_df.index[i], "split"] = "test"

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
        body_df['split'] = row.split
    
        section_df = body_df.groupby('section').agg({
            'text': lambda text_series: ' '.join([text for text in text_series]),
            'title': lambda text_series: [text for text in text_series][0],
            'abstract': lambda text_series: [text for text in text_series][0],
            'cite_spans': lambda cite_spans_series: [cite['ref_id'] for cite_spans in cite_spans_series for cite in cite_spans],
            'split': lambda text_series: [text for text in text_series][0],
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
        section_df = section_df[['paper_id', 'title', 'abstract', 'section', 'text', 'n_bibs', 'n_nonbibs', 'bib_titles', 'bib_abstracts', 'bib_citing_sentences', 'split']]
        return section_df

    section_survey_df = pd.concat(pdf_survey_df.apply(get_section_df, axis=1).values)
    section_survey_df = section_survey_df[section_survey_df["text"].apply(len) >= 1]  # Remove sections without body text
    section_survey_df.to_pickle(os.path.join(args.dataset_path, 'section_survey_df.pkl'))
    logging.info('Done!')


if __name__ == "__main__":
    append_citing_sentence()
    make_section_df()