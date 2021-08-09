from torch.utils.data import Dataset
import torch

class KoElectraClassificationDataset(Dataset):
	def __init__(self,
		device = None,
		tokenizer = None,
		data_list = None,
        label_list = None,
		max_sequence_length = None, # KoBERT max_length
	):

		self.device = device
		self.data = []
		self.tokenizer = tokenizer

		for index, data in enumerate(data_list):
			index_of_words = self.tokenizer.encode(data)

			if len(index_of_words) > max_sequence_length:
				index_of_words = index_of_words[:max_sequence_length]

			token_type_ids = [0] * len(index_of_words)
			attention_mask = [1] * len(index_of_words)

			# Padding Length
			padding_length = max_sequence_length - len(index_of_words)

			# Zero Padding
			index_of_words += [0] * padding_length
			token_type_ids += [0] * padding_length
			attention_mask += [0] * padding_length

			# Label
			label = int(label_list[index])

			self.data.append({
				'input_ids': torch.tensor(index_of_words).to(self.device),
				'token_type_ids': torch.tensor(token_type_ids).to(self.device),
				'attention_mask': torch.tensor(attention_mask).to(self.device),
				'labels': torch.tensor(label).to(self.device)
			})

	def __len__(self):
		return len(self.data)
	def __getitem__(self,index):
		item = self.data[index]
		return item