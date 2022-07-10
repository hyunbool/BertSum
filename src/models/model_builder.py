
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.encoder import TransformerInterEncoder, Classifier, RNNEncoder
from models.optimizers import Optimizer
from transformers import BartModel, BartForConditionalGeneration

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

class Bart(nn.Module):
    def __init__(self, args, temp_dir, load_pretrained_bart, bart_config):
        super(Bart, self).__init__()
        if(load_pretrained_bart):
            #self.model = BartModel.from_pretrained('facebook/bart-base')
            self.model = BartModel.from_pretrained("/home/tako/BertSum/checkpoint-179000")
            
        else:
            self.model = BartModel(bart_config)

    def forward(self, x):
        """segs: segment embedding

        Returns:
            _type_: _description_
        """
        output = self.model(x)
        
        #top_vec = output.last_hidden_state
        top_vec = output.encoder_last_hidden_state
        
        return top_vec


class NVDM(nn.Module):
    '''Implementation of the NVDM model as described in `Neural Variational Inference for
    Text Processing (Miao et al. 2016) <https://arxiv.org/pdf/1511.06038.pdf>`_.
    Args:
        vocab_size (int): The vocabulary size that will be used for the BOW's (how long the BOW
            vectors will be).
        num_topics (:obj:`int`, optional): Set to `100` by default. The number of latent topics
            to maintain. Corresponds to hidden vector dimensionality `K` in the technical writing.
        hidden_size(:obj:`int`, optional): Set to `256` by default. The number of hidden units to
            include in each layer of the multilayer perceptron (MLP).
        hidden_layers(:obj:`int`, optional): Set to `1` by default. The number of hidden layers to
            generate when creating the MLP component of the model.
        nonlinearity(:obj:`torch.nn.modules.activation.*`, optional): Set to
            :obj:`torch.nn.modules.activation.Tanh` by default. Controls which nonlinearity to use
            as the activation function in the MLP component of the model.
    '''
    @staticmethod
    def _param_initializer(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, vocab_size=768, num_topics=50, hidden_size=64, hidden_layers=1, nonlinearity=nn.Tanh):
        super().__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size

        # First MLP layer compresses from vocab_size to hidden_size
        mlp_layers = [nn.Linear(vocab_size, hidden_size), nonlinearity()]
        # Remaining layers operate in dimension hidden_size
        for _ in range(hidden_layers - 1):
            mlp_layers.append(nn.Linear(hidden_size, hidden_size))
            mlp_layers.append(nonlinearity())
        
        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp.apply(NVDM._param_initializer)

        # Create linear projections for Gaussian params (mean & sigma)
        self.mean = nn.Linear(hidden_size, num_topics)
        self.mean.apply(NVDM._param_initializer)

        # Custom initialization for log_sigma
        self.log_sigma = nn.Linear(hidden_size, num_topics)
        self.log_sigma.bias.data.zero_()
        self.log_sigma.weight.data.fill_(0.)

        self.dec_projection = nn.Linear(num_topics, vocab_size)
        self.log_softmax = nn.LogSoftmax(-1)
        
        self.bn = nn.BatchNorm1d(num_topics)
        self.drop = nn.Dropout(0.2)
        
    def encode(self, input_bows):
        pi = self.mlp(input_bows)
        pi = self.drop(pi)
        # Use this to get mean, log_sig for Gaussian
        mean = self.bn(self.mean(pi))
        log_sigma = self.bn(self.log_sigma(pi))
        return mean, log_sigma


    def forward(self, input_bows):
        # Run BOW through MLP
        input_bows = input_bows.squeeze(dim=0)
        # Use this to get mean, log_sig for Gaussian
        mean, log_sigma = self.encode(input_bows)

        # Calculate KLD
        kld = -0.5 * torch.sum(1 - torch.square(mean) +
                               (2 * log_sigma - torch.exp(2 * log_sigma)), 1)
        # kld = mask * kld  # mask paddings

        # Use Gaussian reparam. trick to sample from distribution defined by mu, sig
        # This provides a sample h_tm from posterior q(h_tm | V) (tm meaning topic model)
        epsilons = torch.normal(0, 1, size=(
            input_bows.size()[0], self.num_topics)).to(input_bows.device)
        sample = (torch.exp(log_sigma) * epsilons) + mean
        sample = self.drop(sample)

        return sample

class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, load_pretrained_bart = False, bert_config = None, bart_config = None, is_test=False):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        self.is_test = is_test
        #self.bart_autoencoder = Bart(args, args.temp_dir, load_pretrained_bart, bart_config)
        
        self.bert_autoencoder = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        self.bert_extractor = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        
        self.topicmodel = NVDM(768, num_topics=args.n_topic)
        self.topicmodel.load_state_dict(torch.load("./src/epoch_49.pt"))
        self.topicmodel.eval()
        
        if (args.encoder == 'classifier'):
            self.encoder = Classifier(self.bert_extractor.model.config.hidden_size + args.n_topic)
        elif(args.encoder=='transformer'):
            self.encoder = TransformerInterEncoder(self.bert_extractor.model.config.hidden_size * 2, args.ff_size, args.heads,
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

    def forward(self, x, segs, clss, mask, mask_cls,  rdm_src, rdm_segs, rdm_clss, rdm_mask, rdm_mask_cls, sentence_range=None):

        # # autoencoder
        rdm_top_vec = self.bert_autoencoder(rdm_src, rdm_segs, rdm_mask)
        
        #print(rdm_top_vec.shape)
        rdm_sents_vec = rdm_top_vec[torch.arange(rdm_top_vec.size(0)).unsqueeze(1), rdm_clss]
        rdm_sents_vec = rdm_sents_vec * rdm_mask_cls[:, :, None].float()

        sample = self.topicmodel(rdm_sents_vec)

        # extractor
        top_vec = self.bert_extractor(x, segs, mask)

        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        new_sents_vec = torch.zeros(sents_vec.size(0), sents_vec.size(1), 768 + 20).to(sents_vec.device)   
    
        topic = torch.sum(sample, 0).unsqueeze(0)
        
        # for i in range(sents_vec.size(1)):
        #     dist = torch.abs(torch.sub(sents_vec[:, i], rdm_sents_vec[:,0]))
        #     new_sents_vec[:,i] = torch.cat((sents_vec[:, i], rdm_sents_vec[:,0], dist), 1)
        
        for i in range(sents_vec.size(1)):
            new_sents_vec[:,i] = torch.cat((sents_vec[:, i], topic), dim=1)

        # pooling
        sent_scores = self.encoder(new_sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
