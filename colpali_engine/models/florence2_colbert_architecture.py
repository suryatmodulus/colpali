import torch
from torch import nn
from .florence2_modeling.modeling_florence2 import Florence2ForConditionalGeneration, Florence2PreTrainedModel


class BiFlorence2(Florence2PreTrainedModel):
    def __init__(self, config):
        super(BiFlorence2, self).__init__(config=config)
        self.model: Florence2ForConditionalGeneration = Florence2ForConditionalGeneration(config)
        self.pooling_strategy = "mean"
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        outputs = self.model(*args, **kwargs, decoder_input_ids=kwargs["input_ids"], output_hidden_states=True)
        last_hidden_states = outputs.encoder_last_hidden_state # (batch_size, sequence_length, hidden_size)
        # pooling -mean on attention mask==1
        proj = torch.sum(last_hidden_states * kwargs["attention_mask"].unsqueeze(-1), dim=1) / torch.sum(
            kwargs["attention_mask"], dim=1, keepdim=True
        )
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj


class ColFlorence2(Florence2PreTrainedModel):
    def __init__(self, config):
        super(ColFlorence2, self).__init__(config=config)
        self.model: Florence2ForConditionalGeneration = Florence2ForConditionalGeneration(config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.main_input_name = "doc_input_ids"
        self.post_init()

    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        # outputs = self.model(*args, **kwargs)
        outputs = self.model(*args, **kwargs, decoder_input_ids=kwargs["input_ids"], output_hidden_states=True)
        last_hidden_states = outputs.encoder_last_hidden_state # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj
