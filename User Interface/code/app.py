
from flask import Flask

UPLOAD_FOLDER = 'C:\\Users\\kamal\\OneDrive\\Desktop\\Fake News Detection\\User Interface\\static'

app = Flask(__name__,template_folder='../templates', static_folder='../static')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER