import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import get_activation
from transformers import (
  ElectraModel,
)
from transformers import ElectraConfig

class KoElectraClassificationModel(nn.Module):
    def __init__(self, num_of_classes):
        super(KoElectraClassificationModel, self).__init__()
        
        electra_config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.koelectra_model = ElectraModel.from_pretrained(
            pretrained_model_name_or_path = "monologg/koelectra-base-v3-discriminator",
            config = electra_config,
            num_of_classes = num_of_classes,
        )
        self.classifier_model = nn.Linear(electra_config.hidden_size , num_of_classes)
        self.dropout = nn.Dropout(electra_config.hidden_dropout_prob)
        self.num_of_classes = num_of_classes

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output = self.koelectra_model(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
        )[0]
        output = self.dropout(output[:, 0, :])
        return self.classifier_model(output)
        