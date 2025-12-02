import torch
from torch import nn
from torch.nn import Softmax
from transformers.models.albert.modeling_albert import AlbertLayerGroup
from transformers.modeling_outputs import BaseModelOutput
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
import model_trace

decode_model = 'Albert'  # 'Bert'  'Roberta'  'Albert'
tokenizer = 'albert-base-v2'
victim_model_checkpoint = 'textattack/albert-base-v2-SST-2'
class MyBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.config = config

        self.perturb = None
        self.sofm = Softmax(dim=0)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def p_init(self, input_length, batch_size=1, init_mag=1.0, cuda_device=0, perturb_mask=None):
        SQRT_NUMEL = input_length * self.config.hidden_size
        self.perturb = torch.zeros((batch_size, input_length, self.config.hidden_size),
                                   device=torch.device('cuda:' + str(cuda_device))).uniform_(-init_mag,
                                                                                             init_mag) / SQRT_NUMEL
        self.perturb = torch.mul(perturb_mask.T, self.perturb).detach()
        self.perturb.requires_grad_()

    def p_accu(self, loss, adv_lr=1.0, input_length=None, average_input_length=20, project_to=None, perturb_mask=None):
        if input_length is not None:
            adv_lr *= input_length / average_input_length

        grad = torch.autograd.grad(loss, self.perturb)[0]
        grad = torch.mul(perturb_mask.T, grad).detach()
        grad = (adv_lr * grad / grad.norm()).detach()
        grad = (grad + self.perturb).detach()

        self.perturb = (self.perturb + grad).detach()
        if project_to is not None:
            if self.perturb.norm() > project_to:
                self.perturb = (project_to * self.perturb / self.perturb.norm()).detach()

        self.perturb.grad = None
        self.perturb.requires_grad_()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        if self.perturb is not None:
            hidden_states = hidden_states + self.perturb

        mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class MyAlbertTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])
        self.perturb = None
        self.perturb_pos = -1

    def p_init(self, input_length, batch_size=1, init_mag=1.0, cuda_device=0, perturb_mask=None):
        SQRT_NUMEL = input_length * self.config.hidden_size
        self.perturb = torch.zeros((batch_size, input_length, self.config.hidden_size),
                                   device=torch.device('cuda:' + str(cuda_device))).uniform_(-init_mag,
                                                                                             init_mag) / SQRT_NUMEL
        self.perturb = torch.mul(perturb_mask.T, self.perturb).detach()
        self.perturb.requires_grad_()

    def p_accu(self, loss, adv_lr=1.0, input_length=None, average_input_length=20, project_to=None, perturb_mask=None):
        if input_length is not None:
            adv_lr *= input_length / average_input_length

        grad = torch.autograd.grad(loss, self.perturb)[0]
        grad = torch.mul(perturb_mask.T, grad).detach()
        grad = (adv_lr * grad / grad.norm()).detach()
        self.perturb = (self.perturb + grad).detach()
        if project_to is not None:
            if self.perturb.norm() > project_to:
                self.perturb = (project_to * self.perturb / self.perturb.norm()).detach()

        self.perturb.grad = None
        self.perturb.requires_grad_()

    def set_pos(self, p_index):
        self.perturb_pos = p_index

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i in range(self.config.num_hidden_layers):
            if i == self.perturb_pos and self.perturb is not None:
                hidden_states = hidden_states + self.perturb

            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group: (group_idx + 1) * layers_per_group],
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                mlm_model = AutoModelForMaskedLM.from_pretrained()
                if decode_model == 'Bert':
                    probe_layer = mlm_model.cls
                elif decode_model == 'Roberta':
                    probe_layer = mlm_model.lm_head
                elif decode_model == 'Albert':
                    probe_layer = mlm_model.predictions

                all_attentions = all_attentions + layer_group_output[-1]
                Model_Tracer = model_trace.Prober(tokenizer, probe_layer, 'victim_model', victim_tokenizer='victim_tokenizer',
                                       victim_cuda_device=0, model_cuda_device='GPU：2', decode_mode=decode_model)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + Model_Tracer +(hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )