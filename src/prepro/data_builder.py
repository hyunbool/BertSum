import gc
import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
import time
from os.path import join as pjoin

import torch
from multiprocess import Pool
from pytorch_pretrained_bert import BertTokenizer

from others.logging import logger
from others.utils import clean
from prepro.utils import _get_word_ngrams

from PacSum.code.extractor import *
from PacSum.code.extractor_weighted import *
from PacSum.code.data_iterator import Dataset
from collections import OrderedDict
from transformers import RobertaTokenizer

import h5py

import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            continue
        if (flag):
            tgt.append(tokens)
            flag = False
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", sep_token = '[SEP]', cls_token = '[CLS]', pad_token='[PAD]')
        self.sep_vid = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        self.cls_vid = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        # self.sep_vid = self.tokenizer.vocab['[SEP]']
        # self.cls_vid = self.tokenizer.vocab['[CLS]']
        # self.pad_vid = self.tokenizer.vocab['[PAD]']
        

    def pacsum_weighted(self, sentence_token, extractor):
        # 데이터 가져오기
        tune_dataset = Dataset(sentence_token, vocab_file="./pacssum_models/vocab.txt")
        tune_dataset_iterator = tune_dataset.iterate_once_doc_bert() # tune_dataset_iterator = value
        summaries = extractor.extract_summary(tune_dataset_iterator)

        return summaries[0]
    
    def pacsum(self, sentence_token, extractor):
        # 데이터 가져오기
        tune_dataset = Dataset(sentence_token, vocab_file="./pacssum_models/vocab.txt")
        tune_dataset_iterator = tune_dataset.iterate_once_doc_bert() # tune_dataset_iterator = value
        summaries = extractor.extract_summary(tune_dataset_iterator)

        return summaries[0]
        
        #python preprocess.py -mode format_to_bert -raw_path ./json_data/train -save_path ./bert_data_pacsum_weighted -oracle_mode greedy -n_cpus 1 -log_file ../logs/preprocess.log

    def preprocess_weighted(self, src, tgt, oracle_ids, extractor):
        
        if (len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]
        labels = [0] * len(src)
        for l in oracle_ids:
            labels[l] = 1
        
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        
        labels = [labels[i] for i in idxs]
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]

        if (len(src) < self.args.min_nsents):
            return None
        if (len(labels) == 0):
            return None

        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt]) # gold label
        src_txt = [original_src_txt[i] for i in idxs]
        doc_txt = ' '.join(src_txt)
        
        # rdm
        sents = []

        for i, c in enumerate(cls_ids):
            try:
                sents.append(src_subtoken_idxs[cls_ids[i]:cls_ids[i+1]])
            except IndexError:
                sents.append(src_subtoken_idxs[cls_ids[i]:])
        
        top_idx, scores = self.pacsum(src_txt[:len(sents)], extractor) #sorted(self.pacsum(src_txt[:len(sents)], extractor)) 
        
        #top_idx = [0, 1, 2]

        rdm_labels = F.softmax(torch.Tensor([s[1] for s in scores]), dim=0) #[0 for _ in range(len(labels))]

        # for i in (top_idx):
        #     rdm_labels[i] = 1

        rdm_src = []
        for i in top_idx:
            rdm_src.extend(sents[i])

        # segs
        rdm_segs = []
        flag = 0
        for i in rdm_src:
            rdm_segs.append(flag)
            if i == 102:
                if flag == 0:
                    flag = 1
                elif flag == 1:
                    flag = 0

        # clss
        rdm_clss = []
        
        for i, n in enumerate(rdm_src):
            if n == 101:
                rdm_clss.append(i)
                    

        tgt = sum(tgt, [])
        abstract = ' '.join(tgt)

        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt, doc_txt, rdm_src, rdm_labels, rdm_segs, rdm_clss, abstract
    
    def preprocess(self, src, tgt, oracle_ids, extractor):
        
        if (len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]
        labels = [0] * len(src)
        for l in oracle_ids:
            labels[l] = 1
        
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        
        labels = [labels[i] for i in idxs]
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]

        if (len(src) < self.args.min_nsents):
            return None
        if (len(labels) == 0):
            return None

        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' [SEP] [CLS] '.join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)
        # sep: 50265 / cls: 50267
        
        # src_subtokens = self.tokenizer(text)['input_ids']
        src_subtokens = src_subtokens[:510] # + [50265]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']


        #src_subtoken_idxs = self.tokenizer.tokenize(src_subtokens)
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]

        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]

        labels = labels[:len(cls_ids)]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt]) # gold label
        src_txt = [original_src_txt[i] for i in idxs]
        doc_txt = ' '.join(src_txt)
        
        # rdm
        sents = []

        for i, c in enumerate(cls_ids):
            try:
                sents.append(src_subtoken_idxs[cls_ids[i]:cls_ids[i+1]])
            except IndexError:
                sents.append(src_subtoken_idxs[cls_ids[i]:])
        
        top_idx = sorted(self.pacsum(src_txt[:len(sents)], extractor)) 

        #top_idx = [0, 1, 2]

        rdm_labels = [0 for _ in range(len(labels))]
        for i in (top_idx):
            rdm_labels[i] = 1
            
        rdm_src = []
        for i in top_idx:
            rdm_src.extend(sents[i])
            
        # segs
        rdm_segs = []
        flag = 0
        for i in rdm_src:
            rdm_segs.append(flag)
            if i == 50265:
                if flag == 0:
                    flag = 1
                elif flag == 1:
                    flag = 0
        
        # clss
        rdm_clss = []
        
        for i, n in enumerate(rdm_src):
            if n == 50267:
                rdm_clss.append(i)
                


        tgt = sum(tgt, [])
        abstract = ' '.join(tgt)

        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt, doc_txt, rdm_src, rdm_labels, rdm_segs, rdm_clss, abstract

def format_to_bert_train_weighted(args):
    extractor = PacSumExtractorWithBertWeighted(bert_config_file = "/home/tako/BertSum/pacssum_models/bert_config.json",
                            bert_model_file = "./pacssum_models/pytorch_model_finetuned.bin")
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        print(corpus_type)
        
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        for a in a_lst:
            _format_to_bert_train_weighted(a, extractor)
        # pool = Pool(args.n_cpus)
        # for d in pool.imap(_format_to_bert_train, a_lst, extractor):
        #     pass

        # pool.close()
        # pool.join()
        
def format_to_bert_train(args):
    extractor = PacSumExtractorWithBert(bert_config_file = "/home/tako/BertSum/pacssum_models/bert_config.json",
                            bert_model_file = "./pacssum_models/pytorch_model_finetuned.bin")
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        print(corpus_type)
        
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        for a in a_lst:
            _format_to_bert_train(a, extractor)
        # pool = Pool(args.n_cpus)
        # for d in pool.imap(_format_to_bert_train, a_lst, extractor):
        #     pass

        # pool.close()
        # pool.join()
        
def format_to_bert_test_weighted(args):
    extractor = PacSumExtractorWithBertWeighted(bert_config_file = "/home/tako/BertSum/pacssum_models/bert_config.json",
                            bert_model_file = "./pacssum_models/pytorch_model_finetuned.bin")
    
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        for a in a_lst:
            _format_to_bert_test_weighted(a, extractor)
        # pool = Pool(args.n_cpus)
        # for d in pool.imap(_format_to_bert_train, a_lst, extractor):
        #     pass

        # pool.close()
        # pool.join()
        
def format_to_bert_test(args):
    extractor = PacSumExtractorWithBert(bert_config_file = "/home/tako/BertSum/pacssum_models/bert_config.json",
                            bert_model_file = "./pacssum_models/pytorch_model_finetuned.bin")
    
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        for a in a_lst:
            _format_to_bert_test(a, extractor)
        # pool = Pool(args.n_cpus)
        # for d in pool.imap(_format_to_bert_train, a_lst, extractor):
        #     pass

        # pool.close()
        # pool.join()


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP' ,'-annotators', 'tokenize,ssplit', '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat', 'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def _format_to_bert_train_weighted(params, extractor):
    json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    #logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt'] # tgt: gold label

        if (args.oracle_mode == 'greedy'):
            
            oracle_ids = greedy_selection(source, tgt, 3)
        elif (args.oracle_mode == 'combination'):
            oracle_ids = combination_selection(source, tgt, 3)
            
        b_data = bert.preprocess_weighted(source, tgt, oracle_ids, extractor)
        if (b_data is None):
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt, doc_txt, rdm_src, rdm_labels, rdm_segs, rdm_clss, abstract = b_data
        b_data_dict = {"src": indexed_tokens, "labels": rdm_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, "doc_txt": doc_txt, 'rdm_src': rdm_src,
                       'rdm_segs': rdm_segs, 'rdm_clss': rdm_clss, 'abstract': abstract}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
    
def _format_to_bert_train(params, extractor):
    json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    #logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt'] # tgt: gold label

        if (args.oracle_mode == 'greedy'):
            
            oracle_ids = greedy_selection(source, tgt, 3)
        elif (args.oracle_mode == 'combination'):
            oracle_ids = combination_selection(source, tgt, 3)
            
        b_data = bert.preprocess(source, tgt, oracle_ids, extractor)
        if (b_data is None):
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt, doc_txt, rdm_src, rdm_labels, rdm_segs, rdm_clss, abstract = b_data
        b_data_dict = {"src": indexed_tokens, "labels": rdm_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, "doc_txt": doc_txt, 'rdm_src': rdm_src,
                       'rdm_segs': rdm_segs, 'rdm_clss': rdm_clss, 'abstract': abstract}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
   
def _format_to_bert_test_weighted(params, extractor):
    json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    #logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        #print(d)
        source, tgt = d['src'], d['tgt'] # tgt: gold label

        if (args.oracle_mode == 'greedy'):
            
            oracle_ids = greedy_selection(source, tgt, 3)
        elif (args.oracle_mode == 'combination'):
            oracle_ids = combination_selection(source, tgt, 3)
        b_data = bert.preprocess_weighted(source, tgt, oracle_ids, extractor)
        if (b_data is None):
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt, doc_txt, rdm_src, rdm_labels, rdm_segs, rdm_clss, abstract = b_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": abstract, "doc_txt": doc_txt, 'rdm_src': indexed_tokens,
                       'rdm_segs': segments_ids, 'rdm_clss': cls_ids, 'abstract': abstract}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
    
    
def _format_to_bert_test(params, extractor):
    json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    #logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        #print(d)
        source, tgt = d['src'], d['tgt'] # tgt: gold label

        if (args.oracle_mode == 'greedy'):
            
            oracle_ids = greedy_selection(source, tgt, 3)
        elif (args.oracle_mode == 'combination'):
            oracle_ids = combination_selection(source, tgt, 3)
        b_data = bert.preprocess(source, tgt, oracle_ids, extractor)
        if (b_data is None):
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt, doc_txt, rdm_src, rdm_labels, rdm_segs, rdm_clss, abstract = b_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": abstract, "doc_txt": doc_txt, 'rdm_src': indexed_tokens,
                       'rdm_segs': segments_ids, 'rdm_clss': cls_ids, 'abstract': abstract}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    corpus_mapping = {}
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = f.split('/')[-1].split('.')[0]
        if (real_name in corpus_mapping['valid']):
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            train_files.append(f)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args = params
    source, tgt = load_json(f, args.lower)
    return {'src': source, 'tgt': tgt}



