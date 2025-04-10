from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from data_processing import load_data, NewsDataset
from model import hyperparameter_tuning, train_model, evaluate_model
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
import torch


# 主函数
def main():
    # 加载预训练的 BERT 模型和分词器
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 加载数据
    texts, labels, num_labels = load_data()

    # 超参数调优
    best_params = hyperparameter_tuning(texts, labels, tokenizer, num_labels)
    print('hyperparameter_tuning')

    # 使用最优超参数重新训练模型
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    max_length = 50
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length, augment=True)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': best_params['learning_rate']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': best_params['learning_rate']}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=best_params['learning_rate'])

    train_model(model, train_dataloader, optimizer, device, best_params['epochs'])
    evaluate_model(model, test_dataloader, device)


if __name__ == "__main__":
    main()