import torch
import numpy as np

from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline

from utils import find_valid_answers

output_dir = "weights"
output_dir_trained = output_dir + "/" + "checkpoint-150"


# ============================================================================
# inference (with local trained model)
# ============================================================================
context = "BLOOM has 176 billion parameters and can generate text in 46 natural languages and 13 programming languages."
question = "What can BLOOM do?"
question = "How many programming languages does BLOOM support?"


context = "We find that AFM-on-device is prone to following instructions or answering questions that are present in the input content instead of summarizing it. To mitigate this issue, we identify a large set of examples with such content using heuristics, use AFM-server to generate summaries, as it does not exhibit similar behavior, and add this synthetic dataset to the fine tuning data mixture."
question = "what is the problem?"
question = "how do we fix the problem?"


# A) use pipeline
question_answerer = pipeline("question-answering", model=output_dir_trained)
qa = question_answerer(question=question, context=context)
qa

# B) do non-pipeline training
tokenizer = AutoTokenizer.from_pretrained(output_dir_trained)
inputs = tokenizer(question, context, return_tensors="pt")

model = AutoModelForQuestionAnswering.from_pretrained(output_dir_trained)
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

# TODO: seems to select blank answer
predict_answer_tokens = inputs.input_ids[0,
                                         answer_start_index: answer_end_index + 1]
print(question, tokenizer.decode(predict_answer_tokens))

# ============================================================================
# minimal code, full pipeline, model from HF
# ============================================================================
m = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(m)
model = AutoModelForQuestionAnswering.from_pretrained(m)

question_answerer = pipeline("question-answering", model=m)
# do I need to return offset_mapping? torch doesn't like it when passed to model()
inputs = tokenizer(question, context, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0,
                                         answer_start_index: answer_end_index + 1]
print(question, tokenizer.decode(predict_answer_tokens))

answers = find_valid_answers(inputs, outputs, 4)

for d in answers[:5]:
    score = d['score']
    start = d['start']
    end = d['end']
    print(score)
    predict_answer_tokens = inputs.input_ids[0, start: end + 1]
    print(tokenizer.decode(np.array(predict_answer_tokens.flatten())))

