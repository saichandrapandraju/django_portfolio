import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_question(context, answer):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    model_path = os.path.join(os.path.join(BASE_DIR,'projects'),'qgen_model')

    if 'pytorch_model.bin' not in os.listdir(model_path):
        os.system(f'wget --no-check-certificate https://storage.googleapis.com/text-qgen/pytorch_model.bin -P {model_path}')

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    return tokenizer.batch_decode(model.generate(input_ids=tokenizer(f'qgen answer : {answer} context : {context}', return_tensors='pt').input_ids, num_beams=4), skip_special_tokens=True)[0]