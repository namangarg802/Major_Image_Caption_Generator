from flask import Flask, render_template, url_for, request, redirect
from caption import *
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/generate', methods=['GET'])
def Hi():
    return render_template('caption.html')
@app.route('/generate', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        img = request.files['image']
        print("request ayyi ha")
        print(img)
        print(img.filename)
        img_path='./static/'+img.filename
        print("image ka path"+img_path)
        img.save( img_path)
        caption = caption_this_image(img_path)

        result_dic = {
            'image':  img_path,
            'description': caption
        }
        print(caption)
        
    return render_template('caption.html', results=result_dic)


if __name__ == '__main__':
    app.run(debug=True)
