from functools import partial
import mlx.core as mx
import mlx.nn as nn


def batch_iterate(dataset, batch_size):
    # try to get rid of this dep
    import numpy as np

    perm = np.random.default_rng(12345).permutation(len(dataset))
    # # do not use this??
    # # it won't work at least in juypyter because need ids[iter].item()
    # perm = mx.array(perm)
    for s in range(0, len(dataset), batch_size):
        ids = perm[s: s + batch_size]
        yield dataset[ids]


def loss_fn(model, input_ids, token_type_ids, attention_mask, start_positions,
            end_positions, reduce=True):
    start_logits, end_logits = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        start_positions=start_positions,
        end_positions=end_positions)
    slosses = nn.losses.cross_entropy(start_logits, start_positions)
    elosses = nn.losses.cross_entropy(end_logits, end_positions)
    if reduce:
        slosses = mx.mean(slosses)
        elosses = mx.mean(elosses)
    loss = (slosses + elosses) / 2
    return loss


# TODO review this
def eval_fn(dataset, model, batch_size=8):
    loss = 0
    # refactor with batch_iterate?
    for s in range(0, len(dataset), batch_size):
        batch = dataset[s: s + batch_size]
        input_ids, token_type_ids, attention_mask, start_positions, end_positions = map(
            mx.array,
            (batch['input_ids'], batch['token_type_ids'], batch['attention_mask'],
             batch['start_positions'], batch['end_positions'])
        )
        losses = loss_fn(model, input_ids, token_type_ids, attention_mask,
                         start_positions, end_positions, reduce=False)
        loss += mx.sum(losses).item()
    return loss / len(dataset)


def load_model_tokenizer(hf_model: str, mlx_weights_path: str = "weights/bert-base-uncased.npz"):
    from transformers import AutoConfig, AutoTokenizer
    from model_mlx import BertQA

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    config = AutoConfig.from_pretrained(hf_model)

    model = BertQA(config)
    model.load_weights2(mlx_weights_path)

    return model, tokenizer


def main():
    import mlx.optimizers as optim
    from utils import load_processed_datasets

    print("train or inference using this script")
    print("Follow SETUP IN mlx-examples/lora")

    bert_model = "bert-base-uncased"
    mlx_weights_path = "weights/bert-base-uncased.npz"
    model, tokenizer = load_model_tokenizer(hf_model=bert_model,
                                            mlx_weights_path=mlx_weights_path)

    train_ds, valid_ds, test_ds = load_processed_datasets(
        filter_size=100, model_max_length=tokenizer.model_max_length,
        tokenizer=tokenizer)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(input_ids, token_type_ids, attention_mask, start_positions, end_positions):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(
            model, input_ids, token_type_ids, attention_mask, start_positions, end_positions)
        optimizer.update(model, grads)
    return loss

    optimizer = optim.AdamW(learning_rate=1e-5)

    mx.eval(model.parameters())
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for epoch in range(n_epoch):
        print(f"Training for epoch {epoch+1}...")
        # model.train()
        # for batch in train_dataloader:

        for batch in batch_iterate(train_ds, batch_size=16):
            print(len(batch['input_ids']))

            batch = next(iter(batch_iterate(train_ds, batch_size=16)))

            # input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            # start_positions = batch['start_positions'].to(device)
            # end_positions = batch['end_positions'].to(device)

            # TODO: mx.array() in preprocess_tokenize_function() ????
            input_ids = mx.array(batch['input_ids'])
            # input_ids = mx.expand_dims(input_ids, 0)

            token_type_ids = mx.array(batch['token_type_ids'])
            # token_type_ids = mx.expand_dims(token_type_ids, 0)

            attention_mask = mx.array(batch['attention_mask'])
            # attention_mask = mx.expand_dims(attention_mask, 0)

            start_positions = mx.array(batch['start_positions'])

            end_positions = mx.array(batch['end_positions'])

            outputs, start_logits, end_logits = model(input_ids=input_ids,
                                                      token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask,
                                                      start_positions=start_positions,
                                                      end_positions=end_positions)

            # NEXT: continue here
            # NEXT: MODULARIZE so I can use in Notebook

            # use ipynb notebook for outputs

            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            # # TODO
            loss, _ = loss_and_grad_fn(
                start_logits, end_logits, start_positions, end_positions)
            # end_loss, _ = loss_and_grad_fn(end_logits, end_positions)

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
