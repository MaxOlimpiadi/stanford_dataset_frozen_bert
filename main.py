
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from torch.nn.functional import softmax
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel
import random
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression




#----------------------Global variables----------------------------------------
# Training parameters:
BATCH_SIZE = 16
MAX_LENGTH = 512
EPOCHS = 10
LEARNING_RATE = 5e-5

# Model for getting embeddings
MODEL_NAME = "bert-base-uncased"

# Model (logistic regression):
MAX_ITER = 1000

# Datasets:  
DATASET_FILE = 'stanford_dataset_merged.csv'
EMB_DATASET_FILE = 'emb_dataset.csv'

# Splits:
SPLIT_FOLDER_PATH = 'split'
TRAIN_SLICES_FOLDER_PATH = os.path.join(SPLIT_FOLDER_PATH, 'train_slices')
RANDOM_SEEDS = (7, 10, 35)
TRAIN_SPLIT_SIZES = (20, 40, 70, 100, 150, 200, 300, 500, 700, 900, 1100)

TRAIN_SPLIT_FILE = 'train.csv'
VAL_SPLIT_FILE = 'val.csv'
TEST_SPLIT_FILE = 'test.csv'

# Test prediction results:
TEST_RESULTS_FOLDER = 'test'
TEST_RESULTS_FILE = 'predictions.csv'

# GPU and CUDA:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Experiment:
CREATE_EMBEDDINGS = False
CREATE_SPLITS = False  # Если нужно датасет разбить на трейн, вал, тест. Иначе - сразу грузим все 3 части из соотв. файлов.
LOG_FILE_NAME = "experiments_log.csv"
#------------------------------------------------------------------------------




class HumorDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item



def load_data(path, mode):
    data = pd.read_csv(path)
    if mode == 'texts':
        data["label"] = data["label"].map({"POSITIVE": 1, "NEGATIVE": 0})
        texts = data["text"]
        labels = data["label"]
        split_tags = data['split']
        #breakpoint()
        return texts, labels, split_tags
    elif mode == 'embeddings':
        return data
    else:
        raise ValueError(f'Invalid mode: {mode}')
        




# getting a convinient structure of global train data for further slicing:
def get_samples_by_class(global_train_embeddings, global_train_labels, global_idx, random_state):
    samples_by_class = {
        "positive": [],
        "negative": []
    }
    for embedding, label, idx in zip(global_train_embeddings, global_train_labels, global_idx):
        if label == 1:
            samples_by_class["positive"].append((embedding, label, idx))
        elif label == 0:
            samples_by_class["negative"].append((embedding, label, idx))  
        else: 
            raise ValueError(f"Unexpected label: {label}")
    
    # шафлим:
    random.seed(random_state)
    random.shuffle(samples_by_class["positive"]) 
    random.shuffle(samples_by_class["negative"]) 
    
    return samples_by_class



def create_splits(train_df, test_df):
    
    global_train_embeddings = train_df.loc[:, 'x0':'x767'].to_numpy()
    global_train_idx = train_df['idx'].to_numpy()
    global_train_labels = train_df['label'].to_numpy()

    test_embeddings = test_df.loc[:, 'x0':'x767'].to_numpy()
    test_idx = test_df['idx'].to_numpy()
    test_labels = test_df['label'].to_numpy()

    tmp_embeddings, val_embeddings, tmp_labels, val_labels, tmp_idx, val_idx = train_test_split(
        test_embeddings, 
        test_labels,
        test_idx,
        test_size = 10000,
        shuffle = True,
        random_state = 42,
        stratify = test_labels
    )
    
    reminder_embeddings, test_embeddings, reminder_labels, test_labels, reminder_idx, test_idx = train_test_split(
        tmp_embeddings,
        tmp_labels,
        tmp_idx,
        test_size = 10000,
        shuffle = True,
        random_state = 42,
        stratify = tmp_labels,
    )

    
    for seed in RANDOM_SEEDS:
        seed_folder_path = os.path.join(TRAIN_SLICES_FOLDER_PATH, f'Seed_{seed}')
        samples_by_class = get_samples_by_class(
            global_train_embeddings, 
            global_train_labels,
            global_train_idx, 
            seed
        )
        for size in TRAIN_SPLIT_SIZES:
            train_embeddings, train_labels, train_idx = create_training_slice(samples_by_class, size)
            save_split(
                train_embeddings, 
                train_labels, 
                train_idx, 
                seed_folder_path,
                f'train_{size}.csv'
            )
            
    save_split(val_embeddings, val_labels, val_idx, SPLIT_FOLDER_PATH, VAL_SPLIT_FILE)
    save_split(test_embeddings, test_labels, test_idx, SPLIT_FOLDER_PATH, TEST_SPLIT_FILE)



def create_training_slice(samples_by_class, size):
    n_positive = size // 2
    n_negative = size - n_positive
    positive_slice_tuples = samples_by_class["positive"][:n_positive]
    negative_slice_tuples = samples_by_class["negative"][:n_negative]
    
    current_data = positive_slice_tuples + negative_slice_tuples
    #random.shuffle(current_data)
    
    split_embeddings = []
    split_labels = []
    split_idx = []
    for embedding, label, idx in current_data:
        split_embeddings.append(embedding)
        split_labels.append(label)
        split_idx.append(idx)
            
    return split_embeddings, split_labels, split_idx
    
    
    
def save_split(embeddings, labels, indices, folder_path, filename):
    
    records = []
    for idx, emb, l in zip(indices, embeddings, labels):
        tmp_dict = {
            'idx': idx
        }
        for i, component in enumerate(emb):
            tmp_dict[f'x{i}'] = component
        tmp_dict['label'] = l
        records.append(tmp_dict)            
    
    df = pd.DataFrame(records)
    
    os.makedirs(folder_path, exist_ok = True)
    
    full_path = os.path.join(folder_path, filename)
    df.to_csv(full_path, index = False)
    
  
    
# вернёт список тензоров
def get_cls_embeddings(texts, labels, model, tokenizer, device):
    
    encodings = tokenizer(list(texts), padding = True, truncation = True, max_length = MAX_LENGTH)
    MyDataset = HumorDataset(encodings, labels)
    MyDataLoader = DataLoader(MyDataset, batch_size = BATCH_SIZE)
    
    cls_embeddings = []
    batch_number = 1
    for batch in MyDataLoader:
        print(f'Batch #{batch_number} is being processed...')
        batch = {
            key: value.to(device)
            for key, value in batch.items()
            if key != 'labels'
        }
        with torch.no_grad():
            outputs = model(**batch)
        batch_cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings.extend(batch_cls_embeddings.cpu()) 
        batch_number += 1
    return cls_embeddings



def create_dataloaders(train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE)
    
    return train_dataloader, val_dataloader, test_dataloader


def create_model(model_name):
    print(torch.cuda.is_available())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    return model, tokenizer, device 






def evaluate_model(model, X_test, y_gold):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_gold, y_pred)
    precision = precision_score(y_gold, y_pred)
    recall = recall_score(y_gold, y_pred)
    f1 = f1_score(y_gold, y_pred)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



def save_model(save_path, model, optimizer):
    torch.save(
        {
            "model_state_dict": model.state_dict(),  # Веса модели
            "optimizer_state_dict": optimizer.state_dict(),  # Параметры оптимизатора
        }, 
            save_path
    )
    
    print(f"Model saved to {save_path}")



def save_embeddings_to_file(cls_embeddings, labels, split_tags, EMB_DATASET_FILE):
    results = []
    for idx, (cls_emb, label, s_tag) in enumerate(zip(cls_embeddings, labels, split_tags)):
        row = [idx]
        row.extend(cls_emb.tolist())
        row.append(label)
        row.append(s_tag)
        results.append(row)
    columns = ["idx"] + [f'x{i}' for i in range(len(cls_embeddings[0]))] + ["label"] + ["split"]
    df = pd.DataFrame(results, columns = columns)
    df.to_csv(EMB_DATASET_FILE, index = False)
        
        
        
        


def log_experiment(dataset_name, seed, train_file_name, size, model_name, max_iter, metrics_dict, log_filename = LOG_FILE_NAME):
    row = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_name,
        "seed": seed,
        "train_file_name": train_file_name,
        "size": size,
        "model": model_name,
        "max_iter": max_iter,
        **metrics_dict # Распаковываем наши метрики.
    }
    df = pd.DataFrame([row])
    # Если файл не существует, создаем его с колонками. Если существует — дописываем в конец (mode='a')
    header = not os.path.exists(log_filename)
    df.to_csv(log_filename, mode='a', index=False, header=header)




def plot_metrics(log_file_name):
    data = pd.read_csv(log_file_name)
    # accuracies = data["accuracy"]
    # precisions = data["precision"]
    # recalls = data["recall"]
    # f1_scores = data["f1"]
    # train_sizes = data["size"]
    # seeds = data["seed"]

    grouped = data.groupby('size')
    train_sizes = sorted(data["size"].unique())

    metrics = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1"
    }

    plt.figure(figsize=(10, 6))

    for metric, label in metrics.items():
        means = grouped[metric].mean().reindex(train_sizes)
        stds = grouped[metric].std().reindex(train_sizes)

        plt.plot(
            train_sizes,
            means,
            marker="o",
            linewidth=2,
            label=label
        )

        plt.fill_between(
            train_sizes,
            means - stds,
            means + stds,
            alpha=0.15
        )

    plt.title(
        "Dynamics of metrics depending on the size of the training sample"
    )
    plt.xlabel("Training sample size")
    plt.ylabel("Metric value")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.savefig(
        "my_plot.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    



def main():
    if os.path.exists(LOG_FILE_NAME):
        os.remove(LOG_FILE_NAME) # удаляем весь лог перед новой цепочки экспериментов
    
    if CREATE_EMBEDDINGS:
        data_path =  DATASET_FILE
        texts, labels, split_tags = load_data(data_path, 'texts')
        model, tokenizer, device = create_model(MODEL_NAME)
        cls_embeddings = get_cls_embeddings(texts, labels, model, tokenizer, device)
        save_embeddings_to_file(cls_embeddings, labels, split_tags, EMB_DATASET_FILE)
    
    if CREATE_SPLITS:
        data = load_data(EMB_DATASET_FILE, 'embeddings')
        train_df = data[data['split'] == 'train']
        test_df = data[data['split'] == 'test']
        create_splits(train_df, test_df)
    
    # load test as dataframe and then prepare X and y for logistic regression:
    df_test_data = load_data(os.path.join(SPLIT_FOLDER_PATH, TEST_SPLIT_FILE), 'embeddings')
    X_test = df_test_data.drop(columns = ['idx', 'label']).to_numpy()
    y_test = df_test_data['label'].to_numpy()

    for seed_folder_name in os.listdir(TRAIN_SLICES_FOLDER_PATH):
        print(f'Processing folder: {seed_folder_name}')
        seed_folder_path = os.path.join(TRAIN_SLICES_FOLDER_PATH, seed_folder_name)
        seed = int(seed_folder_name.split('_')[1])
        for slice_file_name in os.listdir(seed_folder_path):
            slice_file_path = os.path.join(seed_folder_path, slice_file_name)
            size = int(
                slice_file_name
                .replace('train_', '')
                .replace('.csv', '')
            )
            print(f'\tslice {slice_file_path}..')

            df_train_data = load_data(slice_file_path, 'embeddings')
            X_train = df_train_data.drop(columns = ['idx', 'label']).to_numpy() 
            y_train = df_train_data['label'].to_numpy()

            model = LogisticRegression(max_iter = MAX_ITER)
            model.fit(X_train, y_train)

            metrics_dict = evaluate_model(model, X_test, y_test)

            log_experiment(DATASET_FILE, seed, slice_file_name, size, 'logistic_regression', MAX_ITER, metrics_dict)

    
    plot_metrics(LOG_FILE_NAME)


    

if __name__ == "__main__":
    main()
    
    
    
    