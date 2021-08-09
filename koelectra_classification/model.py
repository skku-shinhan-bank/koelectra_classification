import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import get_activation
from transformers import (
  ElectraPreTrainedModel,
)
from transformers import ElectraConfig

class KoElectraClassificationModel(nn.Module):
    def __init__(self, num_of_classes):
        super(KoElectraClassificationModel, self).__init__()
        
        electra_config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.pretrained_electra_model = ElectraPreTrainedModel.from_pretrained(
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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        discriminator_hidden_states = self.pretrained_electra_model(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
        )

        output = discriminator_hidden_states[0]
        output = self.dropout(output[:, 0, :])
        return self.classifier_model(output)

class KoElectraClassificationHead(nn.Module):
    def __init__(self, config, num_of_classes):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 4*config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(4*config.hidden_size,num_of_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        