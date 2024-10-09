import torch

from utils import load_squad_raw, split_dataset, preprocess_tokenize_function
from model import load_model_tokenizer_hf


def main(filter_size=500, n_epoch=3, save=False):
    load_split = "train[:" + str(filter_size) + "]"

    squad = load_squad_raw(load_split=load_split, torch=True)
    squad = split_dataset(squad, test_valid_frac=0.2, test_frac=0.5)

    # Define the model
    # This model inherits from PreTrainedModel
    # This model is also a PyTorch torch.nn.Module subclass
    # https://huggingface.co/transformers/v4.9.2/model_doc/bert.html?highlight=bertforquestionanswering

    # Loss? Loss is:
    # Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.

    # also see google BERT run_squad.py
    # eg, compute_loss()
    # https://github.com/google-research/bert/blob/master/run_squad.py
    pre_train_model = 'bert-base-uncased'
    model, tokenizer = load_model_tokenizer_hf(hf_model=pre_train_model)

    args_dict = dict(tokenizer=tokenizer, tensors_kind="pt")
    tokenized_squad = squad.map(preprocess_tokenize_function, batched=True,
                                remove_columns=squad["train"].column_names,
                                fn_kwargs=args_dict)

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_squad['train'], batch_size=16, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(
        tokenized_squad['valid'], batch_size=64)
    test_dataloader = torch.utils.data.DataLoader(
        tokenized_squad['test'], batch_size=64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    device = torch.device("mps")
    model.to(device)

    for epoch in range(n_epoch):
        print(f"Training for epoch {epoch+1}...")
        model.train()
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)

            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total_valid_loss = 0
        for batch in valid_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask,
                                start_positions=start_positions, end_positions=end_positions)
                loss = outputs["loss"]

            total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        print(f'Epoch: {epoch+1}, Valid Loss: {avg_valid_loss}')

    if save:
        p = "weights/pt/fine-tuned-model.pt"
        torch.save(model.state_dict(), p)

    total_test_loss = 0
    for batch in test_dataloader:
        print("Running batch...")
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)
            loss = outputs["loss"]

        total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    print(f'Final Test Loss: {avg_test_loss}')


if __name__ == '__main__':
    main(filter_size=500, n_epoch=3, save=False)
