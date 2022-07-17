from flask import Flask, request, Response ,render_template,send_file,send_from_directory
from werkzeug.utils import secure_filename
from io import BytesIO
from db import db_init, db
from models import *
import cv2
import numpy as np
import base64
from flask import Flask
from flask.globals import request
from flask_socketio import SocketIO
import base64
import numpy as np
import cv2
import os
import facenet_pretrained as facenet
import re
import liveness_model as liveness
#from tensorflow import keras 
package_directory = os.path.dirname(os.path.abspath(__file__))


from collections import defaultdict
q_dict = defaultdict(list)
r_dict = defaultdict(lambda: False)
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(package_directory, './images/')
app.config['SECRET_KEY'] = 'secret!'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
socketio = SocketIO(app, async_handlers=False)
socketio.init_app(app, cors_allowed_origins="*")

facenet_model = facenet.get_model(os.path.join(package_directory, './facenet_weights.h5'))
# SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    
    file = request.files['pic']
    if not file:
        return 'No pic uploaded!', 400
    name = request.form.get('name').replace(' ','_')
    path = os.path.join(app.config['UPLOAD_FOLDER'], f'{name}_{file.filename}')
    file.save(path)
    face_enc = facenet.img_to_encoding(facenet_model, path)
    if face_enc is None:
        return 'No face found!', 400
    np.save(os.path.splitext(path)[0]+'.npy', face_enc)
    person=Person(name=name,path=path)
    db.session.add(person)
    db.session.commit()

    return 'Img Uploaded!', 200

@app.route('/persons')
def get_img():
  x= Person.query.all()
  for person in x :
    print(person.name)
    print(person.path)
  return render_template('persons.html',data=x)
@app.route('/<int:id>')
def get_person_img(id):
    x= Person.query.filter_by(id=id).first()
    if not x:
        return 'Img Not Found!', 404
    return send_file(x.path)

@app.route('/image/<string:path>')
def image(path):
  return send_file(path)

@socketio.on('connect')
def test_connect():
  print(f'{request.sid} connected')

@socketio.on('clear_queue')
def clear_sid():
  sid = request.sid
  if sid in q_dict.keys():
    q_dict.pop(sid)
    r_dict.pop(sid)
  
@socketio.on('disconnect')
def disconnect():
  sid = request.sid
  print(f"{sid} disconnected")
  if sid in q_dict.keys():
    q_dict.pop(sid)
    r_dict.pop(sid)

@socketio.on('check_liveness')
def frame_handler(data):
  sid = request.sid
  frame_bytes = data['frame']
  buff = base64.decodebytes(frame_bytes)
  frame = np.frombuffer(buff, dtype=np.uint8).reshape(200,200,3)
  if not r_dict[sid]:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (100,100), cv2.INTER_AREA)
    frame = cv2.equalizeHist(frame)
    if sid not in q_dict.keys():
      q_dict[sid] = []
    frame_queue = q_dict[sid]
    frame_queue.append(frame)
    if len(frame_queue) >= 24:
      model_inp = np.array(frame_queue, 'uint8').reshape(1,24,100,100,1)
      q_dict[sid] = frame_queue[12:]
      attack_confidence = predict(model_inp, sid)
      if float(attack_confidence) < 0.5:
        r_dict[sid] = True
  else:
    frame = facenet.prepface(frame)
    match = get_id(frame)
    
    if match is None:
      socketio.emit('face_notfound', room=sid)
    else:
      attendance = Day(person_id = match[0])
      db.session.add(attendance)
      db.session.commit()
      socketio.emit('face_found', {'name':match[1]}, room=sid)

def get_id(frame):
  if frame is None:
    return None
  enc = facenet_model.predict(frame).ravel()
  x= Person.query.all()
  dists = []
  names = []
  ids = []
  for person in x:
    ids.append(person.id)
    names.append(person.name)
    enc_path = os.path.splitext(person.path)[0] + '.npy'
    dist = facenet.findCosineDistance(enc, np.load(enc_path))
    dists.append(dist)
  import pdb;pdb.set_trace()
  min_idx = np.argmin(dists)
  if dists[min_idx] < 0.7:
    return ids[min_idx], names[min_idx]
  return None
    

def predict(X, sid):
  sc_d  = liveness.get_scores(X)
  socketio.emit('liveness_score', sc_d, room=sid)
  print(sc_d)
  return sc_d['Attack']
  
  
    
    

if __name__ == '__main__':
  socketio.run(app,debug=True)