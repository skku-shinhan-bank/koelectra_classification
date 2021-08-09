from .model import KoElectraClassificationModel
import torch
from transformers import ElectraTokenizer
from .dataset import KoElectraClassificationDataset
from torch.utils.data import DataLoader

class KoElectraClassificationPredictor:
    def __init__(self, num_of_classes, model_path):
        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(ctx)
        
        model = KoElectraClassificationModel(num_of_classes=num_of_classes)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

        self.classification_model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, data_list, max_sequence_length, batch_size):
        label_list = []

        for data in data_list:
            label_list.append(0)

        dataset = KoElectraClassificationDataset(
            tokenizer=self.tokenizer,
			device=self.device,
			data_list=data_list,
			label_list=label_list,
			max_sequence_length = max_sequence_length,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predict_class_list = []

        for batch_index, data in enumerate(dataloader):
            with torch.no_grad():
                inputs = {
                    'input_ids': data['input_ids'],
                    'attention_mask': data['attention_mask'],
                }
                output = self.classification_model(**inputs)
                max_vals, max_indices = torch.max(output, 1)
                predict_class_list.append(max_indices.item())

        return predict_class_list
                