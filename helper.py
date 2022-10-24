import torch
#Enable GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE=1
#################### split data into train,dev,test##################
def train_dev_test(dataset,random_state=RANDOM_STATE):
    texts=list(dataset["original_text"])
    labels=list(dataset["label"])
    
    target_names = list(set(labels))
    label2idx = {label: idx for idx, label in enumerate(target_names)}
    print(label2idx)

    rest_texts, test_texts, rest_labels, test_labels = train_test_split(texts, labels, test_size=0.1, random_state=RANDOM_STATE)
    train_texts, dev_texts, train_labels, dev_labels = train_test_split(rest_texts, rest_labels, test_size=0.1, random_state=RANDOM_STATE)
    
    print("Train size:", len(train_texts))
    print("Dev size:", len(dev_texts))
    print("Test size:", len(test_texts))
    
    #Create dataframe for coming issue analysis
    df=pd.DataFrame()
    df['original_text']=train_texts+test_texts
    df['label']=train_labels+test_labels
    df['id']=df.index
    return df,(train_texts,dev_texts,test_texts),(train_labels,dev_labels,test_labels),(target_names,label2idx)

import numpy as np
#################### Both class and the following function are used to prepare for input items##################

class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

def convert_examples_to_inputs(example_texts, example_labels, label2idx, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    
    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label2idx[label]

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

        
    return input_items

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

#################### convert data for model input ##################

def get_data_loader(features, max_seq_length, batch_size, shuffle=True): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    #dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    # dataloader tuning in https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
   
    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size,num_workers=2,pin_memory=True)
    return dataloader

from tqdm import trange
from tqdm.notebook import tqdm

def evaluate(model, dataloader):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            #tmp_eval_loss, logits = model(input_ids, attention_mask=input_mask,
            #                              token_type_ids=segment_ids, labels=label_ids)[:2]
            tmp_eval_loss, logits = model(input_ids, attention_mask=input_mask,
                                         labels=label_ids)[:2]  # for distilbert
        outputs = np.argmax(logits.to('cpu'), axis=1)
        label_ids = label_ids.to('cpu').numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
        
    return eval_loss, correct_labels, predicted_labels

import os
from tqdm import trange
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import AdamW,get_linear_schedule_with_warmup


def train(model_name,train_texts,train_labels,dev_texts,dev_labels,target_names,label2idx,params):
    
    
    ## Initialize bert model   
    tokenizer = DistilBertTokenizer.from_pretrained(model_name,target_names=target_names)
    # Using trained model
    model=DistilBertForSequenceClassification.from_pretrained(model_name,num_labels = len(target_names),
                                                             output_attentions = False,
                                                             output_hidden_states = False)  
    
    ## Prepare for data loading and parameter setting for bert model
    train_features = convert_examples_to_inputs(train_texts,train_labels, label2idx, params['MAX_SEQ_LENGTH'], tokenizer)
    train_dataloader = get_data_loader(train_features, params['MAX_SEQ_LENGTH'], params['BATCH_SIZE'], shuffle=True)
    dev_features = convert_examples_to_inputs(dev_texts,dev_labels, label2idx, params['MAX_SEQ_LENGTH'], tokenizer)
    dev_dataloader = get_data_loader(dev_features, params['MAX_SEQ_LENGTH'], params['BATCH_SIZE'], shuffle=True)

    num_train_steps = int(len(train_dataloader.dataset) / params['BATCH_SIZE'] /params['GRADIENT_ACCUMULATION_STEPS'] * params['NUM_TRAIN_EPOCHS'])
    num_warmup_steps = params['NUM_WARMUP_STEPS']

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=params['LEARNING_RATE'], correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,num_training_steps=num_train_steps)
    
    ##Enable GPU if has
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    ## Start to training 
    torch.backends.cudnn.benchmark = True # tuning guide:https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

    loss_history = []
    no_improvement = 0
    PATIENCE=2
    for _ in trange(int(params["NUM_TRAIN_EPOCHS"]), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            #outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids) # non-distillbert
            outputs = model(input_ids, attention_mask=input_mask,labels=label_ids)
            loss = outputs[0]

            if params['GRADIENT_ACCUMULATION_STEPS'] > 1:
                loss = loss / params['GRADIENT_ACCUMULATION_STEPS']

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % params['GRADIENT_ACCUMULATION_STEPS'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),params['MAX_GRAD_NORM'])  

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        dev_loss, _, _ = evaluate(model, dev_dataloader)
            #print("Dev loss:", dev_loss)
    
        print("Loss history:", loss_history)
        print("Dev loss:", dev_loss)

        if len(loss_history) == 0 or dev_loss < min(loss_history):
            no_improvement = 0
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
        else:
            no_improvement += 1

        if no_improvement >= PATIENCE: 
            print("No improvement on development set. Finish training.")
            break


        loss_history.append(dev_loss)
    
from transformers import BertForSequenceClassification,DistilBertForSequenceClassification
from transformers import BertTokenizer,DistilBertTokenizer
import os
from sklearn.metrics import classification_report, precision_recall_fscore_support

OUTPUT_DIR = "./tmp/"
MODEL_FILE_NAME = "pytorch_model.bin"


# Evaluate the dataset based on trained distilbert model
def data_evaluation(texts,labels,model_name,params,trained=True,OUTPUT_DIR = OUTPUT_DIR, MODEL_FILE_NAME = MODEL_FILE_NAME):
    # Convert test data of submission to features
    target_names = list(set(labels))
    label2idx = {label: idx for idx, label in enumerate(target_names)}
    
    # Enable GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Select bert model
    #BERT_MODEL = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
       
    if trained:
        # Using trained model
        model_state_dict = torch.load(os.path.join(OUTPUT_DIR, MODEL_FILE_NAME), map_location=lambda storage, loc: storage)
        model=DistilBertForSequenceClassification.from_pretrained(model_name, state_dict=model_state_dict, num_labels = len(target_names),
                                                                 output_attentions = False,
                                                                 output_hidden_states = False)
    else:
        # Using pretrained model without training
        model=DistilBertForSequenceClassification.from_pretrained(BERT_MODEL,num_labels = len(target_names),
                                                                 output_attentions = False,
                                                                 output_hidden_states = False)        
    model.to(device)
    
    # Convert text and labels to embeddings 
    features = convert_examples_to_inputs(texts, labels, label2idx,  params['MAX_SEQ_LENGTH'], tokenizer)
    dataloader = get_data_loader(features, params['MAX_SEQ_LENGTH'], params['BATCH_SIZE'], shuffle=False)
    
    # Predict the result, and discard the evaluatoin result, only take the prediction result.
    _, correct, predicted = evaluate(model, dataloader)
    print("Errors performance:", precision_recall_fscore_support(correct, predicted, average="micro"))

    bert_accuracy = np.mean(predicted == correct)
    
    #print(bert_accuracy)
    print(classification_report(correct, predicted))

    return correct,predicted, bert_accuracy 

# look for the records with different labels 
def duple_labels(data):
    df_by=pd.DataFrame(data.groupby(['original_text','label']).count().reset_index()[["original_text","label"]])
    df_by=df_by.groupby(by='original_text').count().sort_values('label',ascending=False).reset_index()
    diff_labels=df_by[df_by['label']>1]
    print("Records with different labels: %.2f%%" %(100*len(diff_labels)/len(data)))
    return diff_labels

# Elbow criterion - Determine optimal numbers of clusters by elbow rule.
def elbow_plot(data, maxK=15, seed_centroids=None):
    """
        parameters:
        - data: pandas DataFrame (data to be fitted)
        - maxK (default = 10): integer (maximum number of clusters with which to run k-means)
        - seed_centroids (default = None ): float (initial value of centroids for k-means)
    """
    sse = []
    K= range(1, maxK)
    for k in K:
        if seed_centroids is not None:
            seeds = seed_centroids.head(k)
            kmeans = KMeans(n_clusters=k, max_iter=500, n_init=100, random_state=0, init=np.reshape(seeds, (k,1))).fit(data)
            #data["clusters"] = kmeans.labels_
        else:
            kmeans = KMeans(n_clusters=k, max_iter=300, n_init=100, random_state=0).fit(data)
            #data["clusters"] = kmeans.labels_
        print("k: ", k,"sse: ",kmeans.inertia_)
        # Inertia: Sum of distances of samples to their closest cluster center
        sse.append(kmeans.inertia_)
    plt.figure()
    plt.plot(K,sse,'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    return kmeans.labels_