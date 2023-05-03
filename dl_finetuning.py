from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, AdamW

# 1. Prepare dataset
# 2. Load pretrained Tokenizer, call it with dataset -> encoding
# 3. Build PyTorch Dataset with encodings
# 4. Load pretrained model
# 5. Load Trainer and train it

if __name__ == '__main__': 
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
    model.to('cuda')
    #model.bert.embeddings.requires_grad = True
    #print(model)
    
    text_data_file = "antistereotypes_text.txt"
    labels_data_file = "antistereotypes_labels.txt"
    
    text_list = []
    text_data = open(text_data_file, 'r')
    lines = text_data.readlines()
    for line in lines:
        text_list.append(line)
    #print(len(text_list))

    label_list = []
    label_data = open(labels_data_file, 'r')
    lines = label_data.readlines()
    for line in lines:
        label_list.append(line)
    #print(len(label_list))
    train_data, train_labels, val_data, val_labels = text_list[0:320], label_list[0:320], text_list[320:], label_list[320:]
    train_data = tokenizer(train_data, truncation=True, padding=True)['input_ids']
    train_labels = tokenizer(train_labels, truncation=True, padding=True)['input_ids']
    val_data = tokenizer(val_data, truncation=True, padding=True)['input_ids']
    val_labels = tokenizer(val_labels, truncation=True, padding=True)['input_ids']
    # ignore token_type_ids, it seems wrong and is irrelevant for masked LM anyway
    # taken off HuggingFace website to tune custom dataset
    class AntistereotypeDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {'input_ids': torch.tensor(self.encodings[idx])}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
    train_data = AntistereotypeDataset(train_data, train_labels)
    val_data = AntistereotypeDataset(val_data, val_labels)

    # text = "The developer argued with the designer because [MASK] did not like the design."

    # test_input = tokenizer(text, return_tensors='pt')
    # outputs = model(**test_input)
    # print(outputs)
    # logits = outputs.logits
    # print(logits)

    # mask_token_index = torch.where(test_input["input_ids"] == tokenizer.mask_token_id)[1]
    # print(mask_token_index)

    # mask_token_logits = logits[0, mask_token_index, :]
    # # Pick the [MASK] candidates with the highest logits
    # top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    # for token in top_5_tokens:
    #     print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
    #     """
    #     '>>> The developer argued with the designer because he did not like the design.'
    #     '>>> The developer argued with the designer because they did not like the design.'
    #     '>>> The developer argued with the designer because she did not like the design.'
    #     '>>> The developer argued with the designer because it did not like the design.'
    #     '>>> The developer argued with the designer because developers did not like the design.'
    #     """
    # print(val_data)
    # print(val_data[0])
    

    training_args = TrainingArguments(
        output_dir=f"{model_name}-finetuned",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer
    )
    
    import math

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    def get_grad(module, grad_input, grad_output):
        global input_gradient
        input_gradient = grad_input[0]
    
    model.bert.encoder.layer[0].register_backward_hook(get_grad)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    distill = torch.zeros(16, 25, 768).to('cuda')
    distill_lr = 0.001

    optimizer = AdamW(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            text = batch['input_ids'].to('cuda')
            #print(text.size())
            labels = batch['labels'].to('cuda')
            #print(labels.size())
            outputs = model(text, labels=labels)
            loss = outputs[0]
            # print(loss) # loss reduces
            embedding_output = outputs[2][0]
            #print(embedding_output.size()) # (batch_size (or truncated), seq_len (25 here), 768 embedding size)
            loss.backward()
            # embedding_gradients = model.bert.encoder.layer[0].grad
            # print(embedding_gradients.size()) # torch.Size([30522, 768]) is size of embedding gradients
            encoder_gradients = input_gradient
            distill = distill + (distill_lr * encoder_gradients)
            #print(encoder_gradients.size()) # (16, 25, 768)
            optimizer.step()
    
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    
    model.save_pretrained("models", from_pt=True)

    #print(distill)
    distill_labels = torch.argmax(distill, dim=-1).to('cuda')
    random_mask = torch.rand(distill_labels.size()) < 0.15 # most common masking probability, used in BERT
    random_mask = random_mask.to('cuda')
    mask_token_id = tokenizer.mask_token_id
    temp = torch.full_like(distill_labels, mask_token_id)
    temp.to('cuda')
    print(temp.get_device())
    print(random_mask.get_device())
    print(distill_labels.get_device())
    masked_input_ids = torch.where(random_mask, temp, distill_labels).to('cuda')
    print(masked_input_ids)

    distill_dataset = TensorDataset(masked_input_ids, distill_labels)
    distill_dataloader = DataLoader(distill_dataset, batch_size=16)
    print("distill time")
    distill_trained_model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
    distill_trained_model.to('cuda')
    optimizer = AdamW(distill_trained_model.parameters(), lr=1e-4)
    distill_trained_model.train()
    for epoch in range(3):
        for batch in distill_dataloader:
            optimizer.zero_grad()
            text = batch[0].to('cuda')
            labels = batch[1].to('cuda')
            outputs = distill_trained_model(text, labels=labels)
            loss = outputs.loss
            print(loss)
            loss.backward()
            optimizer.step()
    

    distill_trained_model.save_pretrained("distill_dataset_models", from_pt=True)
    model.save_pretrained("/home/my_name/Desktop/t5small", from_pt=True) 

    # make a new bert model
    # remove embedding layer from it
    # train it on distill once
    # make a trainer with it just like before to evaluate it after adding embedding layer back to it
    # text is distilled data with masks, labels is just distilled data input ids

    #model.eval()
    trainer = Trainer(
        model=distill_trained_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer
    )

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # trainer.train()

    # eval_results = trainer.evaluate()
    # print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # # dataset is a list of dicts, where each dict has 'text' and 'label' 

    # trial = "The developer built a website for the tailor because [MASK] is an expert in building websites."
    # trial = tokenizer(trial, return_tensors='pt')
    # trial.to('cuda')
    # temp = trainer.model(**trial).logits
    # trainer.save_model("./new_model")
    # mask_token_index = torch.where(trial["input_ids"] == tokenizer.mask_token_id)[1]
    # print(mask_token_index)

    # mask_token_logits = temp[0, mask_token_index, :]
    # # Pick the [MASK] candidates with the highest logits
    # top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    # for token in top_5_tokens:
    #     print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
    