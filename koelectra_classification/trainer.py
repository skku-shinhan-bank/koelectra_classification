import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AdamW
from koelectra_classification import KoElectraClassificationModel
import torch
from tqdm import tqdm_notebook
import time
from torch.nn import CrossEntropyLoss
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import ElectraTokenizer
from .dataset import KoElectraClassificationDataset

class KoElectraClassificationTrainer:
	def __init__(self):
		pass

	def train(
		self,
		train_data_list,
		train_label_list,
		test_data_list,
		test_label_list,
		num_of_classes,
		batch_size,
		max_sequence_length,
		learning_rate,
		num_of_epochs,
		max_gradient_normalization,
		warmup_ratio,
		model_output_path,
	):
		ctx = "cuda" if torch.cuda.is_available() else "cpu"
		device = torch.device(ctx)

		classification_model = KoElectraClassificationModel(num_of_classes=num_of_classes).to(device)
		tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

		train_dataset = KoElectraClassificationDataset(
			tokenizer=tokenizer,
			device=device,
			data_list=train_data_list,
			label_list=train_label_list,
			max_sequence_length = max_sequence_length
		)
		test_dataset = KoElectraClassificationDataset(
			tokenizer=tokenizer,
			device=device,
			data_list=test_data_list,
			label_list=test_label_list,
			max_sequence_length = max_sequence_length
		)

		train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
		loss_function = CrossEntropyLoss()
		t_total = len(train_dataloader) * num_of_epochs
		warmup_step = int(t_total * warmup_ratio)
		scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

		# data history for experiments
		history_loss = []
		history_train_acc = []
		history_test_acc = []
		history_train_time = []

		print("[TRAINING]")
		print("train data: {} / train label: {}".format(len(train_data_list), len(train_label_list)))
		print("test data: {} / test label: {}".format(len(test_data_list), len(test_label_list)))
		print("epochs: {}".format(num_of_epochs))
		print("classes: {}".format(num_of_classes))
		print("model output path: {}".format(model_output_path))

		for epoch_index in range(num_of_epochs):
			print("[epoch {}]\n".format(epoch_index + 1))

			train_losses = []
			train_acc = 0.0
			classification_model.train()
			start_time = time.time()
			print('(train)')
			for batch_index, data in enumerate(tqdm_notebook(train_dataloader)):
				optimizer.zero_grad()
				inputs = {
					'input_ids': data['input_ids'],
					'attention_mask': data['attention_mask'],
				}
				output = classification_model(**inputs)
				max_vals, max_indices = torch.max(output, 1)
				train_acc += (max_indices == data['labels']).sum().data.cpu().numpy()
				loss = loss_function(output, data['labels'])
				train_losses.append(loss.item())
				loss.backward()
				torch.nn.utils.clip_grad_norm_(classification_model.parameters(), max_gradient_normalization)
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
			train_time = time.time() - start_time
			train_loss = np.mean(train_losses)
			train_acc = train_acc / len(train_data_list)
			print("acc {} / loss {} / train time {}\n".format(train_acc, train_loss, train_time))
			history_loss.append(train_loss)
			history_train_acc.append(train_acc)
			history_train_time.append(train_time)

			cm = ConfusionMatrix(num_of_classes)
			test_losses = []
			test_acc = 0.0
			classification_model.eval()
			print('(test)')
			for batch_index, data in enumerate(test_dataloader):
				with torch.no_grad():
					inputs = {
						'input_ids': data['input_ids'],
						'attention_mask': data['attention_mask'],
					}
					output = classification_model(**inputs)
					max_vals, max_indices = torch.max(output, 1)
					test_acc += (max_indices == data['labels']).sum().data.cpu().numpy()
					loss = loss_function(output, data['labels'])
					test_losses.append(loss.item())
					for index, real_class_id in enumerate(data['labels']):
						cm.add(real_class_id, max_indices[index].item())
			
			test_loss = np.mean(test_losses)
			test_acc = test_acc / len(test_data_list)
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
			'total_train_step': len(train_dataloader)  # 현재 epoch에 학습 할 총 train step
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
    

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
