import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNCTC(nn.Module):
    def __init__(self, config):
        super(RNNCTC, self).__init__()
        self.config = config
        Encoder = eval(self.config.model.encoder.name)
        Decoder = eval(self.config.model.decoder.name)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, padded_input, input_lengths, padded_target, logit=False):
        """
        Args:
            N is batch_size; 
            Ti is the max number of frames;
            D is feature dim;
            To is max number of transcription symbol
            
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
            encoder_padded_outputs: N x Ti x H (batch_size x seq_length x hidden_size) 
        """
        encoder_padded_inputs, _ = self.encoder(padded_input, input_lengths)
        loss = self.decoder(padded_target, encoder_padded_inputs, input_lengths, logit)
        return loss #loss.mean()
    
    def forward_loss(self, padded_input, input_lengths, padded_target):
        """
        Args:
            N is batch_size; 
            Ti is the max number of frames;
            D is feature dim;
            To is max number of transcription symbol
            
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
            encoder_padded_outputs: N x Ti x H (batch_size x seq_length x hidden_size) 
        """
        encoder_padded_inputs, _ = self.encoder(padded_input, input_lengths)
        loss = self.decoder(padded_target, encoder_padded_inputs, input_lengths)
        return loss
    
    def predict_prob(self, padded_input, input_lengths, padded_target):
        encoder_padded_inputs, _ = self.encoder(padded_input, input_lengths)
        log_probs = self.decoder.predict_prob(padded_target, encoder_padded_inputs, input_lengths)
        return log_probs
    
    def predict_prob_loss(self, padded_input, input_lengths, padded_target):
        encoder_padded_inputs, _ = self.encoder(padded_input, input_lengths)
        probs, loss = self.decoder.predict_prob_loss(padded_target, encoder_padded_inputs, input_lengths)
        return probs, loss

    def get_seq_lens(self, input_length):
        return input_length

class GRUBNEncoder(nn.Module):
    def __init__(self, config):
        super(GRUBNEncoder, self).__init__()
        self.config = config
        self.encoder_config = self.config.model.encoder
        self.rnn = nn.GRU(
            input_size=self.config.feature.input_dim,
            hidden_size=self.encoder_config.hidden_size, 
            num_layers=self.encoder_config.num_layers, 
            dropout=self.encoder_config.dropout,
            bidirectional=self.encoder_config.bidirectional,
            batch_first=True, 
        )
        self.batch_norm = nn.BatchNorm1d(self.encoder_config.hidden_size * 2)

    def forward(self, input_x, enc_len):
        """
        Args:
            N is batch_size; 
            Ti is the max number of frames;
            D is feature dim;
            To is max number of transcription symbol
            
            padded_input: N x Ti x D
            input_lengths: N
        """
        
        total_length = input_x.size(1)  # get the max sequence length

        packed_input = pack_padded_sequence(input_x, enc_len.cpu(), batch_first=True)

        # packed_output: batch_size x seq_length x hidden_size
        # hidden: batch_size, hidden_size
        packed_output, hidden = self.rnn(packed_input)
        
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        output = output.permute(0,2,1)
        # print(f'input size: {output.size()}\t hidden_size: {self.encoder_config.hidden_size}')
        output = self.batch_norm(output).permute(0,2,1)
        # print(f'input size: {output.size()}\t hidden_size: {self.encoder_config.hidden_size}')
        return output, hidden

class CTCDecoder(nn.Module):
    def __init__(self, config):
        super(CTCDecoder, self).__init__()
        self.config = config
        self.decoder_config = self.config.model.decoder

        self.hidden_size = self.decoder_config.hidden_size
        self.vocab_size = self.config.data.vocab_size

        self.ctc_loss = nn.CTCLoss(blank=self.config.data.PAD_token, reduction='none')
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size,
                      self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.vocab_size)
        )
    
    def forward(self, padded_target, padded_input, input_lengths, logit=False):
        
        PAD_token = self.config.data.PAD_token

        # ys = [y[y != PAD_token] for y in padded_input]
        target_lengths = input_lengths.new([len(y[y != PAD_token]) for y in padded_target])
        # print(f'input length{input_lengths.size()}', input_lengths)
        # print(f'target length{target_lengths.size()}', target_lengths)
        mlp_out = self.mlp(padded_input)
        
        log_probs = mlp_out.log_softmax(2).permute(1,0,2)
        # print('mlp size', log_probs.size(), log_probs)
        
        loss = self.ctc_loss(log_probs, padded_target, input_lengths, target_lengths)

        if self.ctc_loss.reduction == 'none':
            loss = loss.div(target_lengths.float())
        # print(loss.size())

        if logit == True:
            return loss, log_probs.transpose(0, 1)
        return loss
    
    def predict_prob(self, padded_target, padded_input, input_lengths):
        PAD_token = self.config.data.PAD_token

        # ys = [y[y != PAD_token] for y in padded_input]
        # target_lengths = input_lengths.new([len(y[y != PAD_token]) for y in padded_target])
        # print(f'input length{input_lengths.size()}', input_lengths)
        # print(f'target length{target_lengths.size()}', target_lengths)
        mlp_out = self.mlp(padded_input)
        
        log_probs = mlp_out.log_softmax(2)
        # probs = mlp_out.softmax(2)
        return log_probs
    
    def predict_prob_loss(self, padded_target, padded_input, input_lengths):
        PAD_token = self.config.data.PAD_token

        # ys = [y[y != PAD_token] for y in padded_input]
        target_lengths = input_lengths.new([len(y[y != PAD_token]) for y in padded_target])
        # print(f'input length{input_lengths.size()}', input_lengths)
        # print(f'target length{target_lengths.size()}', target_lengths)
        mlp_out = self.mlp(padded_input)
        
        log_probs = mlp_out.log_softmax(2)
        log_probs_permute = log_probs.permute(1,0,2)
        # print('mlp size', log_probs.size(), log_probs)
        
        loss = self.ctc_loss(log_probs_permute, padded_target, input_lengths, target_lengths)

        if self.ctc_loss.reduction == 'none':
            loss = loss.div(target_lengths.float())
        # print(loss.size())
        return log_probs.exp(), loss
