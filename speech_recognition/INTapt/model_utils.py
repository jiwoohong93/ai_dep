from transformers import Wav2Vec2Processor, HubertForCTC
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mine(torch.nn.Module):
	def __init__(self, input_size):
		super(Mine, self).__init__()
		self.fc1 = nn.Linear(input_size*2, 512)
		self.fc2 = nn.Linear(512, 100)
		self.fc3 = nn.Linear(100, 1)

	def forward(self, input_1, input_2):
		input = torch.cat((input_1, input_2), axis=1)
		output = F.relu(self.fc1(input))
		output = F.relu(self.fc2(output))
		output = self.fc3(output)
		return output
	
class PromptGeneratorAttention(torch.nn.Module):
	def __init__(self, args, embed_dim, num_heads, dropout, bias=True, do_train=False):
		super(PromptGeneratorAttention, self).__init__()

		self.embed_dim = embed_dim
		self.dropout = dropout
		self.head_dim = embed_dim // num_heads
		self.num_heads = num_heads
		if (self.head_dim * num_heads) != self.embed_dim:
			raise ValueError(
				f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
				f" and `num_heads`: {num_heads})."
			)
		
		self.scaling = self.head_dim**-0.5
		
		self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

		self.training = do_train

	def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
		return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
		
	def forward(self, hidden_states):
		bsz, tgt_len, _ = hidden_states.size()

		# get query proj
		query_states = self.q_proj(hidden_states) * self.scaling
		key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
		value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

		proj_shape = (bsz * self.num_heads, -1, self.head_dim)
		query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
		key_states = key_states.view(*proj_shape)
		value_states = value_states.view(*proj_shape)
		
		src_len = key_states.size(1)
		attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

		if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
			raise ValueError(
				f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
				f" {attn_weights.size()}"
			)
		attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
		attn_output = torch.bmm(attn_probs, value_states)

		if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
			raise ValueError(
				f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
				f" {attn_output.size()}"
			)
			
		attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
		attn_output = attn_output.transpose(1, 2)

		attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

		attn_output = self.out_proj(attn_output)

		return attn_output, attn_weights

class PromptGeneratorFeedForward(torch.nn.Module):
	def __init__(self, args, hidden_size, activation_dropout, hidden_dropout, intermediate_size):
		super(PromptGeneratorFeedForward, self).__init__()

		self.intermediate_dropout = torch.nn.Dropout(activation_dropout)
		self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)

		self.intermediate_act_fn = nn.GELU()

		self.output_dense = nn.Linear(intermediate_size, hidden_size)
		self.output_dropout = nn.Dropout(hidden_dropout)

	def forward(self, hidden_states):
		hidden_states = self.intermediate_dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		hidden_states = self.intermediate_dropout(hidden_states)

		hidden_states = self.output_dense(hidden_states)
		hidden_states = self.output_dropout(hidden_states)
		return hidden_states

class PromptGenerator(nn.Module):
	def __init__(self, args, config):
		super(PromptGenerator, self).__init__()

		self.attention = PromptGeneratorAttention(args, config.hidden_size, config.num_attention_heads, config.attention_dropout)
		self.dropout = nn.Dropout(config.hidden_dropout)
		self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.feed_forward = PromptGeneratorFeedForward(args, config.hidden_size, config.activation_dropout, config.hidden_dropout, config.intermediate_size)
		self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		
		self.prompt_length = args.prompt_length

	def forward(self, hidden_states, attention_mask=None, output_attentions=False):
		attn_residual = hidden_states
		hidden_states = self.layer_norm(hidden_states)
		hidden_states, attn_weights= self.attention(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = attn_residual + hidden_states

		# hidden_states = self.final_layer_norm(hidden_states)
		hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
		# hidden_states = self.final_layer_norm(hidden_states)

		outputs = hidden_states

		if output_attentions:
			outputs += (attn_weights,)

		return outputs[:, :self.prompt_length, :]

class AccentClassifier(nn.Module):
	def __init__(self, args, config, num_labels):
		super(AccentClassifier, self).__init__()

		self.fc1 = nn.Linear(config.hidden_size,768)
		self.fc2 = nn.Linear(768, 512)
		self.fc3 = nn.Linear(512, 256)

		self.output_layer = nn.Linear(256, num_labels) # for l2 arctic = 6 coraal= 7

		self.dropout = nn.Dropout(args.accent_classifier_dropout)
		self.relu = nn.ReLU()

	def forward(self, input_feature):
		hidden_feature = self.relu(self.fc1(input_feature))
		hidden_feature = self.dropout(hidden_feature)
		hidden_feature = self.relu(self.fc2(hidden_feature))
		hidden_feature = self.dropout(hidden_feature)
		accent_feature = self.relu(self.fc3(hidden_feature))
		output_feauture = self.dropout(accent_feature)

		logits = self.output_layer(output_feauture)

		return logits, accent_feature

class AccentRegressor(nn.Module):
	def __init__(self, args):
		super(AccentRegressor, self).__init__()
		self.fc1 = nn.Linear(256, 128)
		self.fc2 = nn.Linear(128,32)
		self.fc3 = nn.Linear(32, 1)

		self.dropout = nn.Dropout(args.accent_classifier_dropout)
		self.relu = nn.ReLU()

	def forward(self, accent_feature):
		hidden_feature = self.relu(self.fc1(accent_feature))
		hidden_feature = self.dropout(hidden_feature)
		hidden_feature = self.relu(self.fc2(hidden_feature))
		hidden_feature = self.dropout(hidden_feature)
		output = self.relu(self.fc3(hidden_feature))

		return output

class AccentModule(nn.Module):
	def __init__(self, args, config, num_labels=6):
		super(AccentModule, self).__init__()
		self.accent_classifier = AccentClassifier(args, config, num_labels)
		self.accent_regressor = AccentRegressor(args)
		self.lamda = args.accent_lamda


	def forward(self, input_feature, asr_loss, batch):
		logits, accent_feature = self.accent_classifier(input_feature)
		accent_intensity = self.accent_regressor(accent_feature)

		return logits, accent_intensity, accent_feature