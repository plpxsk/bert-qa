def load_model_tokenizer_hf(model_str: str = "bert-base-uncased", hf_auto_class="AutoModel"):
    hf_auto_classes = ["AutoModel", "Bert"]
    assert hf_auto_class in hf_auto_classes, f"hf_auto_class must be in {hf_auto_classes}"

    if hf_auto_class == "AutoModel":
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        model = AutoModelForQuestionAnswering.from_pretrained(model_str)
        tokenizer = AutoTokenizer.from_pretrained(model_str)
    elif hf_auto_class == "Bert":
        from transformers import BertTokenizerFast, BertForQuestionAnswering
        model = BertForQuestionAnswering.from_pretrained(model_str)
        tokenizer = BertTokenizerFast.from_pretrained(model_str)

    return model, tokenizer


def load_squad(filter_size=500, test_valid_size=0.2, test_size=0.5, torch=False):
    """Returns HF DatasetDict with train, valid, test components

    Each component has keys:
    dict_keys(['id', 'title', 'context', 'question', 'answers'])
    """
    from datasets import load_dataset, DatasetDict

    # TODO update this to load full squad?
    split_str = "train[:" + str(filter_size) + "]"
    squad = load_dataset("squad", split=split_str)
    if torch:
        squad.set_format("torch")

    squad_train_testvalid = squad.train_test_split(test_size=test_size)
    squad_test_valid = squad_train_testvalid["test"].train_test_split(
        test_size=test_size)
    squad = DatasetDict({
        "train": squad_train_testvalid["train"],
        "valid": squad_test_valid["train"],
        "test": squad_test_valid["test"]
    })

    return squad


def find_context_start_end(sequence_ids):
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1
    return context_start, context_end


def preprocess_tokenize_function(examples, tokenizer, max_length, tensors_kind=None):
    """Adapted from HF

    https://huggingface.co/docs/transformers/en/tasks/question_answering
    """
    # available_tensors = ["mlx", "pt"]
    # assert tensors_kind in available_tensors, f"tensors_kind must be one of {available_tensors}"

    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors=tensors_kind
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        context_start, context_end = find_context_start_end(sequence_ids)

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


def find_valid_answers(inputs, outputs, n_best_size=20):
    import numpy as np

    valid_answers = []

    context_start_index, context_end_index = find_context_start_end(
        inputs.sequence_ids())

    # model provides logits for entire input (question, context, and [SEP], [CLS])
    # so filter only to context
    # HOWEVER, now they are shifted so that index 0 is start of context, not start of input
    # NEED TO shift
    start_logits = outputs.start_logits.flatten().tolist()
    end_logits = outputs.end_logits.flatten().tolist()
    start_logits = start_logits[context_start_index: context_end_index + 1]
    end_logits = end_logits[context_start_index: context_end_index + 1]

    # not more than length of context
    # TODO: move to post-context filter??
    top_k = min(n_best_size, len(start_logits))
    # in plain python:
    topk_start_indices = np.argsort(
        start_logits)[-1: -n_best_size - 1: -1].tolist()
    topk_end_indices = np.argsort(
        end_logits)[-1: -n_best_size - 1: -1].tolist()

    # score all top logits
    for start in topk_start_indices:
        for end in topk_end_indices:
            if start <= end:
                valid_answers.append({
                    "score": start_logits[start] + end_logits[end],
                    # shift indeces back to input-zero'd
                    "start": start + context_start_index,
                    "end": end + context_start_index
                })
    valid_answers.sort(key=lambda x: x['score'], reverse=True)
    return valid_answers
