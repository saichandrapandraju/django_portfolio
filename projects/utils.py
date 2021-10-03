import os
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from keras.models import model_from_json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

qgen_model_path = os.path.join(
    os.path.join(BASE_DIR, 'projects'), 'qgen_model')

traffic_sign_model_path = os.path.join(os.path.join(
    BASE_DIR, 'projects'), 'traffic_sign_model')

en_2_in_model_path = os.path.join(os.path.join(
    BASE_DIR, 'projects'), 'translate_en_2_IN')


def generate_question(context, answer):
    if 'pytorch_model.bin' not in os.listdir(qgen_model_path):
        os.system(
            f'wget --no-check-certificate https://storage.googleapis.com/portfoliomodels/qgen_base.bin --output-document={qgen_model_path}/pytorch_model.bin')
    tokenizer = T5Tokenizer.from_pretrained(qgen_model_path)
    model = T5ForConditionalGeneration.from_pretrained(qgen_model_path)

    return tokenizer.batch_decode(model.generate(input_ids=tokenizer(f'qgen answer : {answer} context : {context}', return_tensors='pt').input_ids, num_beams=4), skip_special_tokens=True)[0]


def predict_traffic_sign(img_path):

    with open(os.path.join(traffic_sign_model_path, 'classes.json')) as f:
        classes = json.load(f)

    height, width, data = 30, 30, []

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
    return classes[str(pred[0]+1)]


def predict_agender(img_path):
    import tensorflow as tf
    import numpy as np
    import cv2
    from PIL import Image
    model_path = os.path.join(os.path.join(
        BASE_DIR, 'projects'), 'traffic_sign_model')

    classes = {1: 'Speed limit (20km/h)',
               2: 'Speed limit (30km/h)',
               3: 'Speed limit (50km/h)',
               4: 'Speed limit (60km/h)',
               5: 'Speed limit (70km/h)',
               6: 'Speed limit (80km/h)',
               7: 'End of speed limit (80km/h)',
               8: 'Speed limit (100km/h)',
               9: 'Speed limit (120km/h)',
               10: 'No passing',
               11: 'No passing veh over 3.5 tons',
               12: 'Right-of-way at intersection',
               13: 'Priority road',
               14: 'Yield',
               15: 'Stop',
               16: 'No vehicles',
               17: 'Veh > 3.5 tons prohibited',
               18: 'No entry',
               19: 'General caution',
               20: 'Dangerous curve left',
               21: 'Dangerous curve right',
               22: 'Double curve',
               23: 'Bumpy road',
               24: 'Slippery road',
               25: 'Road narrows on the right',
               26: 'Road work',
               27: 'Traffic signals',
               28: 'Pedestrians',
               29: 'Children crossing',
               30: 'Bicycles crossing',
               31: 'Beware of ice/snow',
               32: 'Wild animals crossing',
               33: 'End speed + passing limits',
               34: 'Turn right ahead',
               35: 'Turn left ahead',
               36: 'Ahead only',
               37: 'Go straight or right',
               38: 'Go straight or left',
               39: 'Keep right',
               40: 'Keep left',
               41: 'Roundabout mandatory',
               42: 'End of no passing',
               43: 'End no passing veh > 3.5 tons'}

    height = 30
    width = 30
    data = []

    image = cv2.imread(img_path)
    image_from_array = Image.fromarray(image, 'RGB')
    sized_image = image_from_array.resize((height, width))
    data.append(np.array(sized_image))
    model = tf.keras.models.load_model(model_path)

    inp_img = np.array(data)
    inp_img = inp_img.astype('float32')/255
    pred = np.argmax(model.predict(inp_img), axis=-1)
    return classes[pred[0]+1]


def en_2_in(en_txt):

    if 'pytorch_model.bin' not in os.listdir(qgen_model_path):
        os.system(
            f'wget --no-check-certificate https://storage.googleapis.com/portfoliomodels/en_2_in.bin --output-document={en_2_in_model_path}/pytorch_model.bin')

    model = MBartForConditionalGeneration.from_pretrained(en_2_in_model_path)
    tokenizer = MBart50TokenizerFast.from_pretrained(
        en_2_in_model_path, src_lang="en_XX")

    lang_to_key = {
        "Gujarati": "gu_IN",
        "Hindi": "hi_IN",
        "Bengali": "bn_IN",
        "Malayalam": "ml_IN",
        "Marathi": "mr_IN",
        "Tamil": "ta_IN",
        "Telugu": "te_IN"
    }

    model_inputs = tokenizer(en_txt.strip(), return_tensors="pt")
    response = ""
    for k, v in lang_to_key.items():
        generated_tokens = model.generate(
            **model_inputs, forced_bos_token_id=tokenizer.lang_code_to_id[v])
        out = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)[0]
        response += f"{k}: {out}\n"
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
