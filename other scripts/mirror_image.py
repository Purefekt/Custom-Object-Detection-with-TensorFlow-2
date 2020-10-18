import PIL
import os
import os.path
from PIL import Image, ImageOps

f = r'C:\\Users\\Veer\\Desktop\\New\\no_glasses_shirt_MIRROR'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = ImageOps.mirror(img)
    img.save(f_img)