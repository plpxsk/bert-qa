def load_squad(
        load_split: str,
        tokenizer,
        preproc_function,
        test_valid_frac: float,
        test_frac: float,
        return_tuples: bool = False,
        torch: bool = False
):
    """
    Load squad dataset from HuggingFace and preprocess for Q&A modeling
    """
    squad = load_squad_raw(load_split=load_split, torch=torch)
    squad = split_dataset(squad, test_valid_frac=test_valid_frac, test_frac=test_frac)
    squad = preproc_squad(squad, tokenizer=tokenizer, preproc_function=preproc_function,
                          remove_columns=squad["train"].column_names)
    if return_tuples:
        train = squad["train"]
        valid = squad["valid"]
        test = squad["test"]
        return train, valid, test
    else:
        return squad


def load_squad_raw(load_split: str = "train[:100]", torch: bool = False):
    """
    Returns HF Dataset with keys:
    dict_keys(['id', 'title', 'context', 'question', 'answers'])
    """
    from datasets import load_dataset
    squad = load_dataset("squad", split=load_split)
    if torch:
        squad.set_format("torch")
    return squad


def split_dataset(dataset, test_valid_frac: float = 0.25, test_frac: float = 0.5):
    """Split dataset into train, valid, test

    trainining fraction will be 1-test_valid_frac
    test_frac then splits <test_valid_frac>
    """
    from datasets import DatasetDict
    dataset_train_testvalid = dataset.train_test_split(test_size=test_valid_frac)
    dataset_test_valid = dataset_train_testvalid["test"].train_test_split(
        test_size=test_frac)
    dataset = DatasetDict({
        "train": dataset_train_testvalid["train"],
        "valid": dataset_test_valid["train"],
        "test": dataset_test_valid["test"]
    })
    return dataset


def preproc_squad(squad, tokenizer, preproc_function, remove_columns, batched=True):
    args_dict = dict(tokenizer=tokenizer, tensors_kind=None)
    squad = squad.map(preproc_function, batched=batched,
                      remove_columns=remove_columns, fn_kwargs=args_dict)
    return squad


def preprocess_tokenize_function(examples, tokenizer, tensors_kind=None):
    """
    Convert Q&A examples for use in models

    Slightly adapted from HF

    Source B has tokenizer options stride and return_overflowing_tokens, with
    different inequality logic (> start_char, < end_char etc)

    Sources:
    A)
    https://huggingface.co/docs/transformers/en/tasks/question_answering
    https://github.com/huggingface/transformers/blame/main/docs/source/en/tasks/question_answering.md

    B) Nov 22 2022
    https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt#post-processing
    https://github.com/huggingface/course/blame/main/chapters/en/chapter7/7.mdx

    """
    # available_tensors = ["mlx", "pt"]
    # assert tensors_kind in available_tensors, f"tensors_kind must be one of {available_tensors}"

    max_length = tokenizer.model_max_length

    # HF source B: 384 // 3 = 128 for stride
    stride = max_length // 3

    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors=tensors_kind
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        context_start, context_end = find_context_start_end(sequence_ids)

        # If the answer is not fully inside the context, label it (0, 0)
        # NOTE: sources disagree on logic, using more recent source
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
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


def get_answers(start_logits, end_logits, tokenized_input_sequence_ids, flatten_to_list=True,
                top_k=5, n_best_size=20):
    """Get valid answers from Q&A model outputs

    Valid means start_char <= end_char

    Use top_k to return top_k best answers

    Processing starts with n_best_size highest start_logits and end_logits, and
    then sums the two logits for sorting into top_k scores
    """

    context_start_index, context_end_index = find_context_start_end(
        tokenized_input_sequence_ids)

    start_logits = filter_logits_to_context(start_logits, context_start_index, context_end_index,
                                            flatten_to_list=flatten_to_list)
    end_logits = filter_logits_to_context(end_logits, context_start_index, context_end_index,
                                          flatten_to_list=flatten_to_list)

    valid_answers = find_valid_answers(start_logits, end_logits, context_start_index,
                                       n_best_size=n_best_size, sort=True)
    if top_k is not None:
        valid_answers = valid_answers[:top_k]

    return valid_answers


def find_context_start_end(sequence_ids):
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1
    return context_start, context_end


def filter_logits_to_context(logits, context_start_index, context_end_index, flatten_to_list=True):
    """Model provides logits for entire input (question, context, and [SEP],
    [CLS]) so filter only to context

    HOWEVER, now logits are shifted so that index 0 is start of context, not
    start of input. Will need to shift back

    """
    if flatten_to_list:
        logits = logits.flatten().tolist()
    logits = logits[context_start_index: context_end_index + 1]
    return logits


def find_valid_answers(start_logits, end_logits, context_start_index, n_best_size, sort=True):
    import numpy as np

    # can't be more than length of context
    n_best_size = min(n_best_size, len(start_logits))

    n_best_start_indices = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
    n_best_end_indices = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

    # score all top logits
    valid_answers = []

    for start in n_best_start_indices:
        for end in n_best_end_indices:
            if start <= end:
                d = {
                    "score": start_logits[start] + end_logits[end],
                    # ... shift indeces back to input-zero'd
                    "start": start + context_start_index,
                    "end": end + context_start_index
                }
                valid_answers.append(d)
    if sort:
        valid_answers.sort(key=lambda x: x['score'], reverse=True)
    return valid_answers


def init_logger(level):
    import logging

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % level)
    logging.basicConfig(level=numeric_level)
