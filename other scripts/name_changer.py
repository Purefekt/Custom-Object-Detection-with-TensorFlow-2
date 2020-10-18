import os

path = os.chdir("C:\\Users\\Veer\\Desktop\\Dataset\\[1001-2800]\\[2] Dataset Random [1001-2800]")

i = 1001
for file in os.listdir(path):
    
    new_file_name = "{}.jpg".format(i)
    os.rename(file, new_file_name)
    
    i = i+1
    