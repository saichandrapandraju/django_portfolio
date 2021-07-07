import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_question(context, answer):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    model_path = os.path.join(os.path.join(BASE_DIR,'projects'),'qgen_model')

    if 'pytorch_model.bin' not in os.listdir(model_path):
        os.system(f'wget --no-check-certificate https://storage.googleapis.com/portfoliomodels/pytorch_model.bin -P {model_path}')

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    return tokenizer.batch_decode(model.generate(input_ids=tokenizer(f'qgen answer : {answer} context : {context}', return_tensors='pt').input_ids, num_beams=4), skip_special_tokens=True)[0]

def predict_traffic_sign(img_path):
    import tensorflow as tf
    import numpy as np 
    import cv2
    from PIL import Image
    model_path = os.path.join(os.path.join(BASE_DIR,'projects'),'traffic_sign_model')

    classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }

    height = 30
    width = 30
    data=[]

    image=cv2.imread(img_path)
    image_from_array = Image.fromarray(image, 'RGB')
    sized_image = image_from_array.resize((height, width))
    data.append(np.array(sized_image))
    model = tf.keras.models.load_model(model_path)

    inp_img=np.array(data)
    inp_img = inp_img.astype('float32')/255 
    pred = np.argmax(model.predict(inp_img), axis=-1)
    return classes[pred[0]+1]

def predict_agender(img_path):
    import tensorflow as tf
    import numpy as np 
    import cv2
    from PIL import Image
    model_path = os.path.join(os.path.join(BASE_DIR,'projects'),'traffic_sign_model')

    classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }

    height = 30
    width = 30
    data=[]

    image=cv2.imread(img_path)
    image_from_array = Image.fromarray(image, 'RGB')
    sized_image = image_from_array.resize((height, width))
    data.append(np.array(sized_image))
    model = tf.keras.models.load_model(model_path)

    inp_img=np.array(data)
    inp_img = inp_img.astype('float32')/255 
    pred = np.argmax(model.predict(inp_img), axis=-1)
    return classes[pred[0]+1]