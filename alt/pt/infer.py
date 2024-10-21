import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering


def load_model_tokenizer(device):
    pre_train_model = 'bert-base-uncased'
    model = BertForQuestionAnswering.from_pretrained(pre_train_model)
    p = "weights/pt/fine-tuned-model-good.pt"
    model.load_state_dict(torch.load(p, weights_only=True))

    model.eval()

    model.to(device)

    tokenizer = BertTokenizerFast.from_pretrained(pre_train_model)
    return model, tokenizer


if __name__ == '__main__':

    from utils import get_answers

    context = "HF Transformers is backed by the three most popular deep learning libraries - Jax, PyTorch and TensorFlow - with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other"  # noqa
    question = "Which deep learning libraries back HF Transformers?"

    question = "How many programming languages does BLOOM support?"
    context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."  # noqa

    device = "mps"
    model, tokenizer = load_model_tokenizer(device)

    inputs = tokenizer(question, context, return_tensors="pt")
    inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    answers = get_answers(outputs.start_logits, outputs.end_logits, inputs.sequence_ids())
    print("\n", question)
    for d in answers[:5]:
        score = d['score']
        start = d['start']
        end = d['end']
        print(score)
        predict_answer_tokens = inputs.input_ids[0, start: end + 1]
        print(tokenizer.decode(predict_answer_tokens))
