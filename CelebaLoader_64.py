
import os
import struct
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
class CelebaLoader:
    img_width=64
    img_height=64
    def __init__(self, img_folder, label_path):
        self.img_folder=img_folder
        self.label_path=label_path
        self.img_filenames=[]
        self.labels=[]
        self.imgs=[]
        self.load_label()

    def load_label(self):
        attr_file=open(self.label_path,"r")
        attr=attr_file.readlines()[2:]
        for line in attr:
            line=line.split()
            for j in range(1,len(line)):
                line[j]=int(line[j])
                if line[j]==-1:
                    line[j]=0
            self.img_filenames.append(line[0])
            self.labels.append(line[1:])
        self.imgs=[None]*len(self.labels)


    def read(self, batch_size):
        index=np.random.randint(0,len(self.img_filenames),[batch_size])
        retx=[]
        rety=[]
        for i in index:
            if self.imgs[i] is None:
                filename=os.path.join(self.img_folder, self.img_filenames[i])
                img=Image.open(filename)
                img=img.resize([96,96])
                img=np.asarray(img)
                img=img/255.0
                img=img.astype(np.float16)
                img=img[16:80,16:80,:]
                self.imgs[i]=img
            retx.append(self.imgs[i])
            rety.append(self.labels[i])
        return retx,rety

    @staticmethod
    def show(data,rows,cols):
        figure = np.zeros([rows * CelebaLoader.img_width, cols * CelebaLoader.img_height, 3])
        for i in range(rows):
            for j in range(cols):
                figure[i * CelebaLoader.img_width:(i + 1) * CelebaLoader.img_width, j * CelebaLoader.img_height:(j + 1) * CelebaLoader.img_height, :] = data[i * cols + j]
        fig = plt.figure(figsize=(20, 20))
        plt.imshow(figure)
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    IMG_PATH = "./Celeba/img_align_celeba/"
    LABEL_PATH = "./Celeba/list_attr_celeba.txt"
    loader = CelebaLoader(IMG_PATH, LABEL_PATH)
    img,_=loader.read(20)
    CelebaLoader.show(img,4,5)