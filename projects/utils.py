import os, re, json, requests
from transformers import T5ForConditionalGeneration, T5Tokenizer, MBart50TokenizerFast
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from keras.models import model_from_json
import projects.javalang_tokenizer as javalang_tok
from nltk.tokenize import word_tokenize
from fairseq.models.transformer import TransformerModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

qgen_model_path = os.path.join(
    os.path.join(BASE_DIR, 'projects'), 'qgen_model')

traffic_sign_model_path = os.path.join(os.path.join(
    BASE_DIR, 'projects'), 'traffic_sign_model')

en_2_in_model_path = os.path.join(os.path.join(
    BASE_DIR, 'projects'), 'translate_en_2_IN')


def indent_lines(lines):
    prefix = ''
    for i, line in enumerate(lines):
        line = line.strip()
        if re.match('CB_COLON|CB_COMA|CB_', line):
            prefix = prefix[2:]
            line = prefix + line
        elif line.endswith('OB_'):
            line = prefix + line
            prefix += '  '
        else:
            line = prefix + line
        lines[i] = line
    untok_s = '\n'.join(lines)
    return untok_s


def detokenize_java(s):
    assert isinstance(s, str) or isinstance(s, list)
    if isinstance(s, list):
        s = ' '.join(s)
    s = s.replace('ENDCOM', 'NEW_LINE')
    s = s.replace('▁', 'SPACETOKEN')

    s = s.replace('} "', 'CB_ "')
    s = s.replace('" {', '" OB_')
    s = s.replace('*/ ', '*/ NEW_LINE')
    s = s.replace('} ;', 'CB_COLON NEW_LINE')
    s = s.replace('} ,', 'CB_COMA')
    s = s.replace('}', 'CB_ NEW_LINE')
    s = s.replace('{', 'OB_ NEW_LINE')
    s = s.replace(';', '; NEW_LINE')
    lines = re.split('NEW_LINE', s)

    untok_s = indent_lines(lines)
    untok_s = untok_s.replace('CB_COLON', '};').replace(
        'CB_COMA', '},').replace('CB_', '}').replace('OB_', '{')
    untok_s = untok_s.replace('> > >', '>>>').replace('<< <', '<<<')
    untok_s = untok_s.replace('> >', '>>').replace('< <', '<<')

    try:
        # call parser of the tokenizer to find comments and string and detokenize them correctly
        tokens_generator = javalang_tok.tokenize(untok_s, keep_comments=True)
        for token in tokens_generator:
            if isinstance(token, javalang_tok.String) or isinstance(token, javalang_tok.Comment):
                token_ = token.value.replace('STRNEWLINE', '\n').replace('TABSYMBOL', '\t').replace(' ', '').replace(
                    'SPACETOKEN', ' ')
                untok_s = untok_s.replace(token.value, token_)
    except KeyboardInterrupt:
        raise
    except:
        pass
    return untok_s


def generate_question(context, answer):
    try:
        if 'pytorch_model.bin' not in os.listdir(qgen_model_path):
            os.system(
                f'wget --no-check-certificate https://storage.googleapis.com/portfoliomodels/qgen_base.bin --output-document={qgen_model_path}/pytorch_model.bin')
        tokenizer = T5Tokenizer.from_pretrained(qgen_model_path)
        model = T5ForConditionalGeneration.from_pretrained(qgen_model_path)
        output = tokenizer.batch_decode(model.generate(input_ids=tokenizer(f'qgen answer : {answer} context : {context}', return_tensors='pt').input_ids, num_beams=4), skip_special_tokens=True)[0]
    except:
        output = 'Sorry, something went wrong...'
    return output


def predict_traffic_sign(img_path):

    with open(os.path.join(traffic_sign_model_path, 'classes.json')) as f:
        classes = json.load(f)

    height, width, data = 30, 30, []

    try:
        image = cv2.imread(img_path)
        image_from_array = Image.fromarray(image, 'RGB')
        sized_image = image_from_array.resize((height, width))
        data.append(np.array(sized_image))

        with open(os.path.join(traffic_sign_model_path, 'model.json'), 'r') as m_json:
            loaded_model_json = m_json.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(os.path.join(traffic_sign_model_path, "model.h5"))

        inp_img = np.array(data)
        inp_img = inp_img.astype('float32')/255
        pred = np.argmax(model.predict(inp_img), axis=-1)
        output = classes[str(pred[0]+1)]
    except:
        output = 'Sorry, something went wrong...'
    return output


def predict_agender(img_path):
    pass


def en_2_in(en_txt):

    tokenizer = MBart50TokenizerFast.from_pretrained(en_2_in_model_path, src_lang="en_XX")
    API_TOKEN = os.environ.get('HF_API_KEY')
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/facebook/mbart-large-50-one-to-many-mmt"

    def query(payload):
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    lang_to_key = {
        "Gujarati": "gu_IN",
        "Hindi": "hi_IN",
        "Bengali": "bn_IN",
        "Malayalam": "ml_IN",
        "Marathi": "mr_IN",
        "Tamil": "ta_IN",
        "Telugu": "te_IN"
    }

    response = ""
    for k, v in lang_to_key.items():
        try:
            data = query({
                  "inputs": en_txt.strip(),
                  "parameters": {"forced_bos_token_id":tokenizer.lang_code_to_id[v]},
                  })
            response += f"{k}: {data[0]['generated_text']}\n\n"
        except:
            data = query({
                  "inputs": en_txt.strip(),
                  "parameters": {"forced_bos_token_id":tokenizer.lang_code_to_id[v]},
                  "options":{"wait_for_model":True}
                  })
            response += f"{k}: {data[0]['generated_text']}\n\n"
    return response.strip()


def sketch(img_path):
    import cv2
    from PIL import Image

    def dodgeV2(x, y):
        return cv2.divide(x, 255 - y, scale=256)

    try:
        img = cv2.imread(img_path, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_invert = cv2.bitwise_not(img_gray)
        img_smoothing = cv2.GaussianBlur(
            img_invert, (21, 21), sigmaX=0, sigmaY=0)
        final_img = dodgeV2(img_gray, img_smoothing)
        fimg = Image.fromarray(final_img)
        save_path = os.path.join(os.path.join(os.path.join(os.path.join(
            BASE_DIR, 'static'), 'images'), 'sketched'), 'sketch.png')
        os.remove(save_path)
        fimg.save(save_path)
        return save_path
    except Exception as e:
        return None


def ai4code(code, language):
    plbart_source = os.path.join(os.path.join(
        BASE_DIR, 'projects'), 'plbart_source')
    result = ''''''
    if language == 'java':
        paths = [f"{language}_en", f"{language}_cs"]
        for path in paths:
            # print(path)
            model_path = os.path.join(os.path.join(BASE_DIR, 'projects'), path)
            if f'{path}.pt' not in os.listdir(model_path):
                os.system(
                f'wget --no-check-certificate https://storage.googleapis.com/portfoliomodels/{path}.pt --output-document={model_path}/{path}.pt')
            try:
                model = TransformerModel.from_pretrained(
                    model_name_or_path=model_path, checkpoint_file=f'{path}.pt', data_name_or_path=model_path, bpe='sentencepiece', user_dir=plbart_source).eval()
                tok_inp = ' '.join(word_tokenize(code.strip()))
                translated = model.translate(tok_inp, beam=5)
                translated = re.sub(r'`[ ]*`', '"', translated)
                translated = re.sub(r"'[ ]*'", '"', translated)
                if path == 'java_en':
                    result += f"SUMMARY OF YOUR CODE : \n {translated.strip().strip('[en_XX]')}"
                else:
                    result += f"\nTRANSLATION TO C# : \n {detokenize_java(translated).strip()}"
            except Exception as e:
                result = "Sorry, something went wrong..."
        return result
    elif language == 'cs':
        path = f"{language}_java"
        model_path = os.path.join(os.path.join(BASE_DIR, 'projects'), path)
        if f'{path}.pt' not in os.listdir(model_path):
            os.system(
            f'wget --no-check-certificate https://storage.googleapis.com/portfoliomodels/{path}.pt --output-document={model_path}/{path}.pt')
        try:
            model = TransformerModel.from_pretrained(
                model_name_or_path=model_path, checkpoint_file=f'{path}.pt', data_name_or_path=model_path, bpe='sentencepiece', user_dir=plbart_source).eval()
            tok_inp = ' '.join(word_tokenize(code.strip()))
            translated = model.translate(tok_inp, beam=5)
            translated = re.sub(r'`[ ]*`', '"', translated)
            translated = re.sub(r"'[ ]*'", '"', translated)
            result += f"TRANSLATION TO JAVA : \n {detokenize_java(translated).strip()}"
        except Exception as e:
            result = "Sorry, something went wrong..."
        return result
    else:
        path = f"{language}_en"
        model_path = os.path.join(os.path.join(BASE_DIR, 'projects'), path)
        if f'{path}.pt' not in os.listdir(model_path):
            os.system(
            f'wget --no-check-certificate https://storage.googleapis.com/portfoliomodels/{path}.pt --output-document={model_path}/{path}.pt')
        plbart_source = None if path == 'python_en' else os.path.join(
            os.path.join(BASE_DIR, 'projects'), 'plbart_source')
        try:
            model_path = os.path.join(
                os.path.join(BASE_DIR, 'projects'), path)
            model = TransformerModel.from_pretrained(
                model_name_or_path=model_path, checkpoint_file=f'{path}.pt', data_name_or_path=model_path, bpe='sentencepiece', user_dir=plbart_source).eval()
            tok_inp = ' '.join(word_tokenize(code.strip()))
            translated = model.translate(tok_inp, beam=5)
            translated = re.sub(r'`[ ]*`', '"', translated)
            translated = re.sub(r"'[ ]*'", '"', translated)
            result += f"SUMMARY OF YOUR CODE : \n {translated.strip().strip('[en_XX]')}"
        except Exception as e:
            result = "Sorry, something went wrong..."
        return result
