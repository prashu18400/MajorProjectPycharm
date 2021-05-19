import cv2
import numpy as np


def pre_process(crop_img):
    img_ = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    img_ = cv2.resize(img_, (48, 48))
    return img_


def predict(model, crop_img):
    result = []
    image = pre_process(crop_img)
    pred_1 = model.predict(np.array(image / 255))
    sex_f = ['M', 'F']
    age = int(np.round(pred_1[1][0]))
    sex = int(np.round(pred_1[0][0]))
    result.append(age)
    result.append(sex_f[sex])
    return result
