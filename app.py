from flask import Flask, render_template, jsonify, request, Response
import cv2
import requests
import base64
import json                    
import time
import threading
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import datetime
from utils.functions import *
from copy import deepcopy
app = Flask(__name__)

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25',
                    help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true",
                    default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.7,
                    type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4,
                    type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true",
                    default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.8, type=float,
                    help='visualization_threshold')
args = parser.parse_args()


# path = "/home/vkist1/frontend_facerec_VKIST/"
path = "./"

facerec_url = 'https://dohubapps.com/user/daovietanh190499/5000/'

api_list = [facerec_url + 'facerec', facerec_url + 'FaceRec_DREAM', facerec_url + 'FaceRec_3DFaceModeling', facerec_url + 'check_pickup']
api_index = 0
extend_pixel = 50
crop_image_size = 100
predict_labels = []

cropped_face_folder = "facial_images/"

if not os.path.exists(cropped_face_folder):
    os.makedirs(cropped_face_folder)

# ------------VKIST ---------------
# vkist_6
# secret_key = '13971a9f-1b2d-46bb-b829-d395431448fd'

# ----------- HHSC -----------------------
# hhsc_3
secret_key = "6c24a661-7bc6-4c28-b057-8c4919285205"

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def loadBase64Img(uri):
    encoded_data = uri.split(',')[1]
    # nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def load_image(img):
	exact_image = False; base64_img = False; url_img = False
	if type(img).__module__ == np.__name__:
		exact_image = True

	elif len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True

	elif len(img) > 11 and img.startswith("http"):
		url_img = True
	#---------------------------

	if base64_img == True:
		img = loadBase64Img(img)

	elif url_img:
		img = np.array(Image.open(requests.get(img, streaming=True).raw))

	elif exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")

		img = cv2.imread(img)
	return img

def face_recognize(frame, isTinyFace=False):
    global api_index, predict_labels

    _, encimg = cv2.imencode(".jpg", frame)
    img_byte = encimg.tobytes()
    img_str = base64.b64encode(img_byte).decode('utf-8')
    new_img_str = "data:image/jpeg;base64," + img_str
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'charset': 'utf-8'}
    payload = json.dumps({"secret_key": secret_key, 'local_register' : isTinyFace, "img": new_img_str,})

    response = requests.post(api_list[api_index], data=payload, headers=headers, timeout=100)
    try:
        for id, name, profileID, timestamp in zip( 
                                                response.json()['result']['id'],
                                                response.json()['result']['identities'],
                                                response.json()['result']['profilefaceIDs'],
                                                response.json()['result']['timelines']
                                                ):
            print('Server response', response.json()['result']['identities'])
            if id != -1:
                # response_time_s = time.time() - seconds
                # print("Server's response time: " + "%.2f" % (response_time_s) + " (s)")
                cur_profile_face = None

                if profileID is not None:
                    cur_url = facerec_url + 'images/' + secret_key + '/' + profileID

                frame = cv2.resize(frame, (crop_image_size, crop_image_size))
                _, encimg = cv2.imencode(".jpg", frame)
                img_byte = encimg.tobytes()
                img_str = base64.b64encode(img_byte).decode('utf-8')
                new_img_str = "data:image/jpeg;base64," + img_str

                now = round(datetime.datetime.now().timestamp() * 1000)
                cv2.imwrite(cropped_face_folder + str(now) + '.jpg', frame)

                predict_labels.append([id, name, new_img_str, cur_profile_face, timestamp, cur_url])

    except requests.exceptions.RequestException:
        print(response.text)

def face_detection_cctv(image):

    img = np.float32(image)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    # Forward pass
    loc, conf, landms = net(img) 

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(
        0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(
        0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    return np.concatenate((dets, landms), axis=1).astype(int)

def get_frame_0():
    global final_img, last_time
    while True:
        if len(final_img) > 0:
            # Convert the frame to a jpeg image
            ret, jpeg = cv2.imencode('.jpg', final_img)

            # Return the image as bytes
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            final_img = cv2.imread("./utils/no_cameras_found.png")
@app.route('/streaming')
def streaming():
    return Response(get_frame_0(), mimetype = 'multipart/x-mixed-replace; boundary=frame')


@app.route('/img_detection', methods=['POST'])
def img_detection():
    global raw_img, final_img, last_time
    queue = []
    
    if len(raw_img) > 0 and (datetime.datetime.now().timestamp() - last_time) < 1:
        final_img = deepcopy(raw_img)
        frame_width = raw_img.shape[1]
        frame_height = raw_img.shape[0]
    
        temp_boxes  = face_detection_cctv(raw_img)
        # print('Detect face in:', str(new_frame_time_0 - prev_frame_time), ' ms')
        for index, b in enumerate(temp_boxes):
            b = list(map(int, b))

            xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            local_reg = False
            w = xmax - xmin
            h = ymax - ymin
            
            # if w < 100 or h < 100:
            #     local_reg = True
            #     ratio = 0.1

            #     xmin -= int(w*ratio)
            #     xmax += int(w*ratio) 
            #     ymin -= int(h*ratio)
            #     ymax += int(h*ratio)
                 
            extend_pixel = 100
            xmin -= extend_pixel
            xmax += extend_pixel 
            ymin -= extend_pixel
            ymax += extend_pixel

            xmin = 0 if xmin < 0 else xmin
            ymin = 0 if ymin < 0 else ymin
            xmax = frame_width if xmax >= frame_width else xmax
            ymax = frame_height if ymax >= frame_height else ymax

            queue = [t for t in queue if t.is_alive()]
            if len(queue) < 3:

                queue.append(threading.Thread(target=face_recognize, args=(raw_img[ymin:ymax, xmin:xmax], local_reg,)))
                queue[-1].start()
                
        draw_box(final_img, temp_boxes, color=(125, 255, 125))

        # _, encimg = cv2.imencode(".jpg", final_img)
        # img_byte = encimg.tobytes()
        # img_str = base64.b64encode(img_byte).decode('utf-8')
        # new_img_str = "data:image/jpeg;base64," + img_str

        return jsonify({"result": {'message': 'Streaming images!', 'img': 'new_img_str'}}), 200
    else:
        final_img = cv2.imread("./utils/no_cameras_found.png")
        return jsonify({"result": {'message': 'Facial image is not ready!', 'img': ''}}), 400
         
@app.route('/img_uploading', methods=['POST'])
def img_uploading():
    global raw_img, last_time

    req = request.get_json()

    new_frame= ""
    if "img" in list(req.keys()):
        new_frame = req["img"]

    validate_img = False
    if len(new_frame) > 11 and new_frame[0:11] == "data:image/":
        validate_img = True

    if validate_img != True:
        return jsonify({"result": {'message': 'Vui lòng truyền ảnh dưới dạng Base64'}}), 400
    
    raw_img = load_image(new_frame)

    last_time = datetime.datetime.now().timestamp()
    return jsonify({"result": {'message': 'Image upload successfully!'}}), 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    global predict_labels
    if len(predict_labels) > 10:
        predict_labels = predict_labels[-10:]
    newest_data = list(reversed(predict_labels))
    return jsonify({"result": {'message': 'Success', 'data': newest_data}}), 200

# @app.route('/data', methods=['POST'])
# def data():
#     global predict_labels
#     if len(predict_labels) > 3:
#         predict_labels = predict_labels[-3:]
#     newest_data = list(reversed(predict_labels))
#     return jsonify({"result": {'message': 'Success', 'data': newest_data}}), 200

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    # elif args.network == "resnet50":
    #     cfg = cfg_re50
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    cudnn.benchmark = True
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    net = net.to(device)

    raw_img = ""
    final_img = ""
    last_time = datetime.datetime.now().timestamp()
    app.run(host='0.0.0.0', debug=True, port=6001)

