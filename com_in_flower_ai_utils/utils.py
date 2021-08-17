import base64

def decodeImage(imgstring, Filename):
    imgdata = base64.b64decode(imgstring)
    with open(Filename, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImageintoBase64(croppedImagePath):
    with open(croppedImagePath, 'rb') as f:
        return base64.b64encode(f.read())