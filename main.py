import telebot, tf_keras as keras  # Импортируем tf-keras - это совместимая версия Keras для работы с .h5 моделями
from tf_keras.models import load_model  # Загружаем функцию load_model из tf_keras, чтобы открыть модель
from PIL import Image, ImageOps  # Installing pillow instead of PIL
import numpy as np
bot = telebot.TeleBot("")



def get_class(img_path, mod_path="bot2/keras_model.h5", lbl_path="bot2/labels.txt"):
    np.set_printoptions(suppress=True)
    model = load_model(mod_path, compile=False)
    class_names = open(lbl_path, "r", encoding="UTF-8").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(img_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:],confidence_score

def get_info(cl):
    if cl == 'дорога\n':
        return "Дорога — обустроенная полоса земли для движения транспорта и пешеходов."
    elif cl == 'пешеходка\n':
        return"Пешеходная — зона или улица, предназначенная только для пешеходов."
    elif cl == 'велодорожка\n':
        return"Велодорожка — полоса или отдельная трасса для движения велосипедистов."

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, f'Привет! Я бот !')
    print(message)



@bot.message_handler(content_types=['photo'])
def send_photo1(msg):
    print(msg.photo)
    file_info = bot.get_file(msg.photo[-1].file_id)
    file_name = file_info.file_path.split('/')[-1]
    downloaded_file = bot.download_file(file_info.file_path)
    with open("bot2/" + file_name, 'wb') as new_file:
        new_file.write(downloaded_file)
    bot.reply_to(msg, f'Я скачал твое фото!')
    name, veroyatnost = get_class("bot2/" + file_name)
    bot.reply_to(msg, 
                 f'Ваша картинка обработана. На изображений: {name}С вероятностью{veroyatnost}')
    bot.reply_to(msg, get_info(name))

bot.infinity_polling(allowed_updates=['message', 'message_reaction'])
