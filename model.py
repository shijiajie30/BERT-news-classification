import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import logging
from data_processing import NewsDataset

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 训练模型
def train_model(model, train_dataloader, optimizer, device, epochs):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        logging.info(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}')


# 评估模型
def evaluate_model(model, test_dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    logging.info(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
    return accuracy, precision, recall, f1


# 超参数调优
def hyperparameter_tuning(texts, labels, tokenizer, num_labels):
    param_grid = {
        'batch_size': [8, 16],
        'learning_rate': [1e-5],
        'epochs': [2]
    }
    best_accuracy = 0
    best_params = None

    for params in ParameterGrid(param_grid):
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2,
                                                                              random_state=42)
        max_length = 50  # 由于新闻文本较长，适当增加最大长度
        train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length, augment=True)
        test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length)

        train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

        model_name = 'bert-base-uncased'
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # 分层学习率
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01, 'lr': params['learning_rate']},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': params['learning_rate']}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=params['learning_rate'])

        train_model(model, train_dataloader, optimizer, device, params['epochs'])
        accuracy, _, _, _ = evaluate_model(model, test_dataloader, device)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    logging.info(f'Best hyperparameters: {best_params}, Best accuracy: {best_accuracy}')
    return best_params