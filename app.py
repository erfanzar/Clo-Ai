import cv2 as cv
import torch
from flask import Flask, render_template
from skimage import io
from torchvision.transforms import Resize, ToPILImage, ToTensor, Normalize

from models.configs import v_config_C
from models.model import Network

IMAGE_SIZE = 224

app = Flask(__name__)

ADDRESS = '192.168.1.103'

# classes = None
# if classes is None:
#     with open(f'{os.getcwd()}\\classes.yaml', 'r') as read:
#         classes = yaml.full_load(read)

model_type = Network(config=v_config_C)
model_material = Network(config=v_config_C)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_material.load_state_dict(torch.load('.\\Detect-Models\\model_material.pt', map_location=DEVICE))
model_type.load_state_dict(torch.load('.\\Detect-Models\\model_type.pt', map_location=DEVICE), )

BASE_URL = 'https://img.mytheresa.com/1088/1088/66/jpeg/catalog/product/6b/'

dict_material = {'Not sure': 0, 'thread': 1, 'Canvas / rubber / plastics': 2, 'Denim': 3, 'Poplin': 4,
                 'cotton / wool': 5, 'Cotton / Linen': 6, 'Other': 7, 'UnKnown-Top': 8, 'Polyester ': 9, 'fabrics': 10,
                 'UnKnown-Body': 11, ' polyester / cotton': 12, 'cotton / linen': 13, 'Polyester / straw': 14,
                 'polyester / rayon': 15, 'Georgette / Crêpe': 16, 'jeans / tartan': 17, 'Skip': 18,
                 'Worsted Wool / Flannel / Fresco': 19}

dict_labels = {'Not sure': 0, 'T-Shirt': 1, 'Shoes': 2, 'Shorts': 3, 'Shirt': 4, 'Pants': 5, 'Skirt': 6, 'Other': 7,
               'Top': 8, 'Outwear': 9, 'Dress': 10, 'Body': 11, 'Long-sleeve': 12, 'Undershirt': 13, 'Hat': 14,
               'Polo': 15, 'Blouse': 16, 'Hoodie': 17, 'Skip': 18, 'Blazer': 19}


@app.route('/')
def home():
    return '<a>init page <a>'


@app.route('/<image>', methods=['GET', 'POST'])
def predictor(image):
    url = f'{BASE_URL}/{image}'
    image = io.imread(url)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    image = torch.from_numpy(image).type(torch.FloatTensor)
    if image.shape[0] > 3:
        image = image.reshape(image.shape[2], image.shape[1], image.shape[0])
    image = Normalize((0, 0, 0), (1, 1, 1))(image)
    image = ToPILImage()(image)
    image = Resize((IMAGE_SIZE, IMAGE_SIZE))(image)
    image = ToTensor()(image)
    image = image.to(DEVICE)
    image = image.reshape(1, 3, 224, 224)
    model_material.eval()
    model_type.eval()
    y_material = model_material.forward(image)
    y_type = model_type.forward(image)
    y_type = torch.argmax(y_type, dim=1)
    y_material = torch.argmax(y_material, dim=1)
    list_material_classes = list(dict_material)
    list_type_classes = list(dict_labels)
    predicts = {
        'predict_num_0': {
            'Type': list_type_classes[y_type],
            'Material': list_material_classes[y_material]
        },
        'predict_num_1': {
            'Type': list_type_classes[y_type],
            'Material': list_material_classes[y_type]
        },
        'predict_num_2': {
            'Type': list_type_classes[y_material],
            'Material': list_material_classes[y_material]
        },
        'show_predict': {
            'Type': list_type_classes[y_type],
            'Material': list_material_classes[y_material]
        }

    }
    return render_template('main.html', predict=predicts)


if __name__ == "__main__":
    app.run(debug=True, host=ADDRESS)
