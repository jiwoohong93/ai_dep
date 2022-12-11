import torch

class ImportanceGetter:
    def __init__(self, model, optimizer, criterion, tokenizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.tokenizer= tokenizer
        
    def get_layers_importance(self, inputs, att_masks, labels, model, optimizer):
        # Back-Propagation
        model.train()
        outputs = self.model(input_ids=inputs, attention_mask=att_masks, labels=labels)
        loss = outputs.loss
        
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        # Compute Importance
        impts = {}
        for name, param in model.named_parameters():
            impts[name] = param * param.grad
            
        return impts
    
    def get_word_pieces_importance(self, input_pieces_ids, word_embed_impts):
        vocab_impts = torch.sum(word_embed_impts, dim=-1)
        input_word_pieces_impts = vocab_impts[input_pieces_ids]
        return input_word_pieces_impts
    
    def detect_word_cluster(self, input_pieces_ids):
        word_start_idxs = []
        word_piece_tokens = self.tokenizer.convert_ids_to_tokens(input_pieces_ids)
        
        for i, word_piece_token in enumerate(word_piece_tokens):
            if word_piece_token[0] == "▁":
                word_start_idxs.append(i)
                
        return word_start_idxs
    
    def cluster_word_importance(self, input_word_pieces_ids, input_word_pieces_impts):
        cluster_impts = []
        detected_cluster = self.detect_word_cluster(input_word_pieces_ids)
        detected_cluster = detected_cluster+[len(input_word_pieces_ids)]
        
        for i in range(len(detected_cluster)-1):
            cluster = input_word_pieces_impts[detected_cluster[i]:detected_cluster[i+1]]
            cluster_impts.append(torch.sum(cluster).item())
            
        clustered_words = self.tokenizer.decode(input_word_pieces_ids).split(' ')
        
        return cluster_impts, clustered_words