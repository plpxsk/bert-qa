def batch_iterate(examples, batch_size):
    perm = np.random.default_rng(12345).permutation(len(examples))
    perm = mx.array(perm)
    for s in range(0, len(examples), batch_size):
        ids = perm[s: s + batch_size]
        yield examples[ids]


def main():
    import mlx.core as mx
    import mlx.optimizers as optim

    from model import BertQA
    from utils import load_squad, preprocess_tokenize_function

    print("train or inference using this script")
    print("Follow SETUP IN mlx-examples/lora")

    bert_model = "bert-base-uncased"
    mlx_weights_path = "weights/bert-base-uncased.npz"
    # follow load_model explicitly for BertQA
    # model, tokenizer = load_model(bert_model, mlx_weights_path)
    config = AutoConfig.from_pretrained(bert_model)
    model = BertQA(config)
    model.load_weights2(mlx_weights_path)

    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    # # for Bert()
    # batch = ["This is an example of BERT working on MLX."]
    # tokens = tokenizer(batch, return_tensors="mlx", padding=True)
    # output, pooled = model(**tokens)

    squad = load_squad(filter_size=100, torch=False)

    # CONTINUE
    max_length = tokenizer.model_max_length
    # NOTE: mlx kind. UPDATE: no
    args_dict = dict(tokenizer=tokenizer, tensors_kind=None,
                     max_length=max_length)

    # batched=False for mlx tensors_kind
    squad_tokenized = squad.map(preprocess_tokenize_function, batched=False,
                                remove_columns=squad["train"].column_names,
                                fn_kwargs=args_dict)

    train_ds = squad_tokenized["train"]
    valid_ds = squad_tokenized["valid"]
    test_ds = squad_tokenized["test"]

    # train_dataloader = torch.utils.data.DataLoader(
    #     tokenized_squad['train'], batch_size=16, shuffle=True)
    # valid_dataloader = torch.utils.data.DataLoader(
    #     tokenized_squad['valid'], batch_size=64)
    # test_dataloader = torch.utils.data.DataLoader(
    #     tokenized_squad['test'], batch_size=64)

    # Define the optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer = optim.AdamW(learning_rate=1e-5)

    # pytorch
    # device = torch.device("mps")
    # model.to(device)

    # right?
    # following https://ml-explore.github.io/mlx/build/html/examples/mlp.html
    mx.eval(model.parameters())

    # NEXT
    # batch = train_ds[2]
    # run code line by line...
    # ... need to implement BertQA():
    # outputs = model(input_ids, attention_mask=attention_mask,
    # ...                             start_positions=start_positions, end_positions=end_positions)

    # Traceback (most recent call last):
    #   File "/Users/paul/github/qa/qa.py", line 73, in <module>
    #     end_positions = batch['end_positions'].to(device)
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # TypeError: Bert.__call__() got an unexpected keyword argument 'start_positions'

    for epoch in range(n_epoch):
        print(f"Training for epoch {epoch+1}...")
        # model.train()
        # for batch in train_dataloader:
        for batch in batch_iterate(train_ds, batch_size=16):
            # input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            # start_positions = batch['start_positions'].to(device)
            # end_positions = batch['end_positions'].to(device)

            # TODO: mx.array() in preprocess_tokenize_function() ????
            input_ids = mx.array(batch['input_ids'])
            input_ids = mx.expand_dims(input_ids, 0)

            token_type_ids = mx.array(batch['token_type_ids'])
            token_type_ids = mx.expand_dims(token_type_ids, 0)

            attention_mask = mx.array(batch['attention_mask'])
            attention_mask = mx.expand_dims(attention_mask, 0)

            start_positions = batch['start_positions']
            end_positions = batch['end_positions']

            outputs, logits = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)

            # loss from nn.value_and_grad() or similar??
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model
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

    # Now, check TEST score
    total_test_loss = 0
    for batch in test_dataloader:
        print(f"Running batch...")
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

    if args.train:
        opt = opt
        train()
        mx.savez(save_file)

    if args.test:
        model.eval()
        test_loss = test_loss
        test_ppl = test_ppl

        return test_loss, test_ppl

    if args.inference:
        context, question = context, question
        run()


def train():
    pass


def evaluate():
    pass


def build_parser():
    parser = argparse.ArgumentParser(description="TBD")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )


if __name__ == '__main__':
    main()
