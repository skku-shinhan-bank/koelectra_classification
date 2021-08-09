import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AdamW
from koelectra_classification import KoElectraClassificationModel
import torch
from tqdm import tqdm_notebook
import time

class KoElectraClassificationTrainer:
	def __init__(self):
		pass

	def train(
		self,
		train_data,
		train_label,
		test_data,
		test_label,
		num_of_classes,
		batch_size,
		max_sequence_length,
		learning_rate,
		num_of_epochs,
		device,
		model_output_path
	):
		classification_model = KoElectraClassificationModel(num_of_classes=num_of_classes).to(device)
		tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

		train_zipped_data = make_zipped_data(train_data, train_label)
		test_zipped_data = make_zipped_data(test_data, test_label)

		train_dataset = KoElectraClassificationDataset(tokenizer=tokenizer, device=device, zipped_data=train_zipped_data, max_seq_len = max_sequence_length)
		test_dataset = KoElectraClassificationDataset(tokenizer=tokenizer, device=device, zipped_data=test_zipped_data, max_seq_len = max_sequence_length)

		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{
				'params': [p for n, p in classification_model.named_parameters() if not any(nd in n for nd in no_decay)],
				'weight_decay': 0.01
			},
			{
				'params': [p for n, p in classification_model.named_parameters() if any(nd in n for nd in no_decay)],
				'weight_decay': 0.0
			},
		]
		optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

		# data history for experiments
		history_loss = []
		history_train_acc = []
		history_test_acc = []
		history_train_time = []

		for epoch_index in range(num_of_epochs):
			print("[epoch {}]\n".format(epoch_index + 1))

			train_losses = []
			train_acc = 0
			classification_model.train()
			start_time = time.time()
			print('(train)')
			for batch_index, data in enumerate(tqdm_notebook(train_loader)):
				optimizer.zero_grad()
				inputs = {
					'input_ids': data['input_ids'],
					'attention_mask': data['attention_mask'],
					'labels': data['labels']
				}
				outputs = classification_model(**inputs)
				loss = outputs[0]
				logit = outputs[1]
				train_losses.append(loss.item())
				loss.backward()
				optimizer.step()
				train_acc += (logit.argmax(1)==inputs['labels']).sum().item()
			end_time = time.time()
			train_loss = np.mean(train_losses)
			train_acc = train_acc / len(train_dataset)
			print("acc {} / loss {} / time {}\n".format(train_acc, train_loss, end_time - start_time))
			history_loss.append(train_loss)
			history_train_acc.append(train_acc)
			history_train_time.append(end_time - start_time)

			cm = ConfusionMatrix(num_of_classes)
			test_losses = []
			test_acc = 0
			classification_model.eval()
			print('(test)')
			for batch_index, data in enumerate(test_loader):
				with torch.no_grad():
					inputs = {
						'input_ids': data['input_ids'],
						'attention_mask': data['attention_mask'],
						'labels': data['labels']
					}
					outputs = classification_model(**inputs)
					loss = outputs[0]
					logit = outputs[1]
					test_losses.append(loss.item())
					test_acc += (logit.argmax(1)==inputs['labels']).sum().item()
					
					for index, real_class_id in enumerate(inputs['labels']):
						cm.add(real_class_id.item(), logit.argmax(1)[index].item())
			
			test_loss = np.mean(test_losses)
			test_acc = test_acc / len(test_dataset)
			print("acc {} / loss {}".format(test_acc, test_loss))
			print("<confusion matrix>\n", pd.DataFrame(cm.get()))
			print("\n")
			history_test_acc.append(test_acc)

		torch.save({
			'epoch': num_of_epochs,  # 현재 학습 epoch
			'model_state_dict': classification_model.state_dict(),  # 모델 저장
			'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
			'loss': loss.item(),  # Loss 저장
			'train_step': num_of_epochs * batch_size,  # 현재 진행한 학습
			'total_train_step': len(train_loader)  # 현재 epoch에 학습 할 총 train step
		}, model_output_path)

		# Print the result
		print("RESULT - copy and paste this to the report")
		for epoch_index in range(num_of_epochs):
			print('epoch ', epoch_index, end='\t')
			print('')
		for i in history_loss:
			print(i, end='\t')
			print('')
		for i in history_train_acc:
			print(i, end='\t')
			print('')
		for i in history_test_acc:
			print(i, end='\t')
			print('')
		for i in history_train_time:
			print(i, end='\t')
			print('')

def make_zipped_data(data, label):      
	zipped_data = []

	for i in range(len(data)):
		row = []
		row.append(data[i])
		row.append(label[i])
		zipped_data.append(row)

	return zipped_data

class KoElectraClassificationDataset(Dataset):
	def __init__(self,
		device = None,
		tokenizer = None,
		zipped_data = None,
		max_seq_len = None, # KoBERT max_length
	):

		self.device = device
		self.data =[]
		self.tokenizer = tokenizer

		for zd in zipped_data:
			index_of_words = self.tokenizer.encode(zd[0])

			if len(index_of_words) > max_seq_len:
				index_of_words = index_of_words[:max_seq_len]

			token_type_ids = [0] * len(index_of_words)
			attention_mask = [1] * len(index_of_words)

			# Padding Length
			padding_length = max_seq_len - len(index_of_words)

			# Zero Padding
			index_of_words += [0] * padding_length
			token_type_ids += [0] * padding_length
			attention_mask += [0] * padding_length

			# Label
			label = int(zd[1])
			data = {
				'input_ids': torch.tensor(index_of_words).to(self.device),
				'token_type_ids': torch.tensor(token_type_ids).to(self.device),
				'attention_mask': torch.tensor(attention_mask).to(self.device),
				'labels': torch.tensor(label).to(self.device)
			}

			self.data.append(data)

	def __len__(self):
		return len(self.data)
	def __getitem__(self,index):
		item = self.data[index]
		return item

def calc_accuracy(X,Y):
	max_vals, max_indices = torch.max(X, 1)
	train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
	return train_acc

class ConfusionMatrix:
  def __init__(self, column):
    self.matrix = []
    for i in range(column):
      self.matrix.append([])
      for j in range(column):
       self. matrix[i].append(0)
  def add(self, real_class_id, predict_class_id):
    self.matrix[real_class_id][predict_class_id] += 1
  def show(self):
    for row in self.matrix:
      print(row)
  def get(self):
    return self.matrix
    
