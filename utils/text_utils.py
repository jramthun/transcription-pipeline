import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "grammarly/coedit-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
intruction = "Fix grammatical errors and punctuation in this sentence:"

def correct_text(text: str | list):
    """
    Correct grammatical errors and punctuation in a given text or list of texts.

    Args:
    text (str | list): The text or list of texts to be corrected.

    Returns:
    str | list: The corrected text or list of corrected texts.

    Raises:
    TypeError: If the input is not a str or list.
    """
    if isinstance(text, str):
        model_input = f"{intruction} {text.strip()}"
        input_ids = tokenizer(model_input, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=256)
        edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return edited_text

    # TODO: find a more memory efficient way of doing this -> this isn't bad though
    elif isinstance(text, list):
        model_input = [f"{intruction} {t['text'].strip()}" for t in text]

        batch_size = 128
        tokenizer.pad_token = tokenizer.eos_token

        batched_outputs = []
        for i in (batched_input := tqdm(range(0, len(model_input), batch_size))):
            batched_input.set_description("Correcting text")
            model_inputs = tokenizer(model_input[i: i + batch_size], return_tensors="pt", padding=True).to(device)
            outputs = model.generate(**model_inputs, max_length=256)
            edited_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for _, edited in enumerate(edited_text):
                batched_outputs.append(edited)

        return batched_outputs

    else:
        raise TypeError("Text is invalid type. Expected str or list.")
