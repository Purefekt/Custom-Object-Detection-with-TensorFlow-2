import PIL
import os
import os.path
from PIL import Image

f = r'C:\\Users\\Veer\\Desktop\\1'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((720,720))
    img = img.rotate(270)
    img.save(f_img)