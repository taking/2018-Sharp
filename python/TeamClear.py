from flask import Flask, request, render_template, redirect, url_for, send_from_directory  # 웹 모듈
# from flask_uploads import UploadSet, configure_uploads, AUDIO, patch_request_class
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

import os, traceback
import PIL
from PIL import Image
import subprocess as sp
import simplejson
from lib.upload_file import uploadfile

import random
import string  # random String 생성 모듈
#import tensorflow as tf

########### Configuration ###########
FFMPEG_BIN = "ffmpeg"
ALLOWED_EXTENSIONS = set(['gif', 'png', 'jpg', 'jpeg', 'bmp', 'mp4', 'mkv', 'wmv'])
IGNORED_FILES = set(['.DS_Store'])

app = Flask(__name__)
app.config['SECRET_KEY'] = 'TeamClear1234'
app.config['UPLOAD_FOLDER'] = '_uploads/'
app.config['DOWNLOAD_FOLDER'] = '_downs/'
app.config['THUMBNAIL_FOLDER'] = '_uploads/thumbnail/'
# app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

bootstrap = Bootstrap(app)
#######################################


############# Function ###############
def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def random_filename(length):
  letters = string.ascii_uppercase
  return ''.join(random.choice(letters) for i in range(length))
def duplicate_file(filename):
  i = 1
  while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
    name, extension = os.path.splitext(filename)
    filename = '%s_%s%s' % (name, str(i), extension)
    i += 1
  return filename
def create_thumbnail(image):
  try:
    base_width = 80
    img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], image))
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), PIL.Image.ANTIALIAS)
    img.save(os.path.join(app.config['THUMBNAIL_FOLDER'], image))
    return True

  except:
    print(traceback.format_exc())
    return False
#######################################


############# HTTP method ##############
@app.route('/', methods=['GET', 'POST'])
def index():
  return render_template('index.html')


@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    files = request.files['file']
    if files:
      filename = secure_filename(files.filename)
      filename = duplicate_file(filename)
      mime_type = files.content_type
      #return redirect(url_for('.multi', uuid=randomword(3)))

    if not allowed_file(files.filename):
      result = uploadfile(name=filename, type=mime_type, size=0, not_allowed_msg="File type not allowed")
      #return 'file uploaded successfully'
      
    else:
      # save file to disk
      uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      files.save(uploaded_file_path)
      # create thumbnail after saving
      if mime_type.startswith('image'):
        create_thumbnail(filename)
      # get file size after saving
      size = os.path.getsize(uploaded_file_path)
      # return json for js call back
      result = uploadfile(name=filename, type=mime_type, size=size)
    return simplejson.dumps({"files": [result.get_file()]})
  if request.method == 'GET':
    # get all file in ./upload directory
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'],f)) and f not in IGNORED_FILES ]
      
    file_display = []

    for f in files:
        size = os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], f))
        file_saved = uploadfile(name=f, size=size)
        file_display.append(file_saved.get_file())

    return simplejson.dumps({"files": file_display})

  return redirect(url_for('index'))

@app.route("/delete/<string:filename>", methods=['DELETE'])
def delete(filename):
  file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  file_thumb_path = os.path.join(app.config['THUMBNAIL_FOLDER'], filename)

  if os.path.exists(file_path):
      try:
          os.remove(file_path)

          if os.path.exists(file_thumb_path):
              os.remove(file_thumb_path)
          
          return simplejson.dumps({filename: 'True'})
      except:
          return simplejson.dumps({filename: 'False'})


# serve static files
@app.route("/thumbnail/<string:filename>", methods=['GET'])
def get_thumbnail(filename):
  return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename=filename)


@app.route("/data/<string:filename>", methods=['GET'])
def get_file(filename):
  return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER']), filename=filename)


if __name__ == '__main__':
    #app.run(host='0.0.0.0')
    app.run(debug=True)     # Debugger is active!