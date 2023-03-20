import os
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from transformers import RobertaTokenizer, RobertaModel
from torch.nn.init import xavier_uniform_

from models.encoder import TransformerInterEncoder, Classifier, RNNEncoder
from models.optimizers import Optimizer
from transformers import BartModel, BartForConditionalGeneration
from torch.distributions import LogNormal, Dirichlet, kl_divergence

from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pickle
import gc


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics=768, id2word=None, hidden_size=64, hidden_layers=2, nonlinearity=nn.Softplus):
        super().__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.id2word = id2word

        # First MLP layer compresses from vocab_size to hidden_size
        mlp_layers = [nn.Linear(vocab_size, hidden_size), nonlinearity()]
        for _ in range(hidden_layers - 1):
            mlp_layers.append(nn.Linear(hidden_size, hidden_size))
            mlp_layers.append(nonlinearity())

        self.mlp = nn.Sequential(*mlp_layers)
        # using Dirichlet distribution directly
        self.h2t = nn.ModuleList([nn.Linear(hidden_size, num_topics),
                                 nn.BatchNorm1d(num_topics)])
        self.mean = nn.Linear(hidden_size, num_topics)
                                  
        self.log_sigma = nn.Sequential(nn.Linear(hidden_size, num_topics),
                                       nn.BatchNorm1d(num_topics))
                                       
        
        self.dec_projection = nn.Linear(num_topics, vocab_size)
        self.log_softmax = nn.LogSoftmax(-1)
        self.softmax = nn.Softmax(-1)
        
        self.bn = nn.BatchNorm1d(num_topics)
        self.drop = nn.Dropout(0.2)
    def encode(self, input_bows):
        h = self.mlp(input_bows)
        h = self.drop(h)
        return h

    def reparameterize(self, h):
        # print(h.shape)
        # print(h)
        alpha = self.h2t[1](self.h2t[0](h)).exp()

        posterior = Dirichlet(alpha.cpu())
        
        # if self.training:
        s = posterior.rsample().cuda()
        # else:
        #     s = posterior.mean().cuda()

        return s, posterior

    def decode(self, sample):
        return self.dec_projection(sample)

    def forward(self, input_bows):
        h = self.encode(input_bows)
        sample, posterior = self.reparameterize(h)
        
        # mean = self.mean(h)
        # log_sigma = self.log_sigma(h)
        # epsilons = torch.normal(0, 1, size=(
        # input_bows.size()[0], self.num_topics)).to(input_bows.device)
        
        # sample = (torch.exp(log_sigma) * epsilons) + mean
        sample = self.drop(sample)
        logits = self.softmax(self.decode(sample))
        rec_loss = -1 * torch.sum(logits * input_bows, 1)

        
        alphas = torch.ones_like(posterior.concentration)
        prior = Dirichlet(alphas)
        
        # kld = F.kl_div(sample, epsilons, size_average=True)
        #kld = kl_divergence(posterior, prior).cuda()
        return sample#, logits, kld, rec_loss

class Bert(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, bert_config = None):
        super(Bert, self).__init__()
        if(load_pretrained_bert):
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel(bert_config)
    def forward(self, x, segs, mask):
        """segs: segment embedding

        Returns:
            _type_: _description_
        """
        encoded_layers, _ = self.model(x, segs, attention_mask=mask)
        top_vec = encoded_layers[-1]
        return top_vec

class RoBerta(nn.Module):
    def __init__(self, temp_dir):
        super(RoBerta, self).__init__()
        #self.model = BartModel.from_pretrained('facebook/bart-base')
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.model.resize_token_embeddings(len(self.tokenizer)+3)
        #self.model = BartModel.from_pretrained("/home/tako/BertSum/checkpoint-179000")
            

    def forward(self, x):
        """segs: segment embedding
        Returns:
            _type_: _description_
        """
        output = self.model(input_ids=x)
        
        #top_vec = output.last_hidden_state
        top_vec = output.last_hidden_state
        
        return top_vec
    
        
class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, load_pretrained_bart = False, bert_config = None, bart_config = None, is_test=False):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        self.is_test = is_test
        self.n_topic = self.args.n_topic

        self.bert_extractor = Bert(args.temp_dir, load_pretrained_bert, bert_config)

        
        self.threshold_dict = {10: 0.0971650390340262, 30: 0.030141125277372813, 50: 0.017866419390398635, 100: 0.008823595488367579, 150: 0.0058770036856582945, 200: 0.0044062418769848745}
        
        with open('./ldamodel_%s.pkl' % str(self.args.n_topic), 'rb') as f:
            self.ldamodel = pickle.load(f)
        
        with open('./corpus.pkl', 'rb') as f:
            self.dictionary = pickle.load(f)
        
        self.threshold = self.threshold_dict[self.args.n_topic]


        if (args.encoder == 'classifier'):
            self.encoder = Classifier(self.bert_extractor.model.config.hidden_size)#+ args.n_topic)
        elif(args.encoder=='transformer'):
            self.encoder = TransformerInterEncoder(self.bert_extractor.model.config.hidden_size, args.ff_size, args.heads,
                                                   args.dropout, args.inter_layers)
        elif(args.encoder=='rnn'):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
                                      input_size=self.bert_extractor.model.config.hidden_size * 2, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
        elif (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert_extractor.model.config.vocab_size, hidden_size=args.hidden_size,
                                     num_hidden_layers=6, num_attention_heads=8, intermediate_size=args.ff_size)
            
            self.bert_extractor.model = BartModel(bert_config)
            self.bert_autoencoder.model = BertModel(bert_config)
            
            self.encoder = Classifier(self.bert_extractor.model.config.hidden_size * 2)

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        
        self.to(device)
        
    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, rdm_src, rdm_segs, rdm_clss, rdm_mask, rdm_mask_cls, sents):#, tp, sentence_range=None):
        
        # autoencoder
        #rdm_top_vec = self.bert_extractor, rdm_segs, rdm_mask)
        rdm_top_vec = self.bert_extractor(rdm_src)#.detach().to('cpu')#.numpy()#, rdm_segs, rdm_mask)

        rdm_sents_vec = rdm_top_vec[torch.arange(rdm_top_vec.size(0)).unsqueeze(1), rdm_clss]
        rdm_sents_vec = rdm_sents_vec * rdm_mask_cls[:, :, None].float()

        if rdm_sents_vec.shape[0] == 1: # batch 1
            rdm_sents_vec = rdm_sents_vec.squeeze(0)
            sample = self.topicmodel(rdm_sents_vec) # [3, 20]
            
            topic = torch.mean(sample, 0).unsqueeze(0) # [1, 20]
        else: # batch 2
            #topic = torch.zeros(rdm_sents_vec.size(0), rdm_sents_vec.size(1), self.n_topic).to(rdm_sents_vec.device)   
            topic = torch.zeros(rdm_sents_vec.size(0), rdm_sents_vec.size(1), 768).to(rdm_sents_vec.device)   
            for i in range(rdm_sents_vec.shape[0]):
                topic[i] = self.topicmodel(rdm_sents_vec[i])
            topic = torch.mean(topic, 1)
        
        
        # extractor
        # top_vec = self.bert_extractor(x, segs, mask) # 모든 문서에 대한 임베딩 [문장개수, 768]
        top_vec = self.bert_extractor(x) # 모든 문서에 대한 임베딩 [문장개수, 768]

        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()


        new_sents_vec = torch.zeros(sents_vec.size(0), sents_vec.size(1), 768).to(sents_vec.device)   

        
        # for i in range(sents_vec.size(1)):
        #     dist = torch.abs(torch.sub(sents_vec[:, i], rdm_sents_vec[:,0]))
        #     new_sents_vec[:,i] = torch.cat((sents_vec[:, i], rdm_sents_vec[:,0], dist), 1)
        
        # topic = torch.Tensor(topic.to('cpu'))
        # # concat
        # for i in range(sents_vec.size(1)):   
        #     sent = torch.Tensor(sents_vec[:, i].to('cpu'))
        #     new_sents_vec[:,i] = torch.cat((topic, sent), dim=1)


        # mul
        topic = torch.Tensor(topic.to('cpu'))
        for i in range(sents_vec.size(1)):
            sent = torch.Tensor(sents_vec[:, i].to('cpu'))
            new_sents_vec[:,i] = torch.mul(sent, topic).to(self.device)#, dim=1) # i번째 문장 + 토픽분포 = [788]

        
        # add
        # topic = torch.Tensor(topic.to('cpu'))
        # for i in range(sents_vec.size(1)):
        #     sent = torch.Tensor(sents_vec[:, i].to('cpu'))
        #     new_sents_vec[:,i] = torch.add(sent, topic).to(self.device)#, dim=1) # i번째 문장 + 토픽분포 = [788]


        # pooling
        sent_scores = self.encoder(new_sents_vec, mask_cls).squeeze(-1)
        
        return sent_scores, mask_cls


