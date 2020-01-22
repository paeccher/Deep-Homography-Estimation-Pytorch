import torch.utils as utils
import torch
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
import random
import cv2
import os

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]

def noisy(noise_typ, img):
    if noise_typ == "gauss":
        mean = 0
        var = 10
        
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (240, 320)) #  np.zeros((224, 224), np.float32)

        noisy_image = np.zeros(img.shape, np.float32)

        if len(img.shape) == 2:
            noisy_image = img + gaussian
        else:
            noisy_image[:, :, 0] = img[:, :, 0] + gaussian
            noisy_image[:, :, 1] = img[:, :, 1] + gaussian
            noisy_image[:, :, 2] = img[:, :, 2] + gaussian

        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        noisy_image = noisy_image.astype(np.uint8)
        return noisy_image

    elif noise_typ == "s&p":
        prob=0.1
        output = np.zeros(img.shape,np.uint8)
        thres = 1 - prob 
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        return output

    elif noise_typ == "compress":
        img = cv2.imread(img,0)
        cv2.imwrite('tmp.jpg', img,  encode_param)
        output = cv2.imread('tmp.jpg',0)
        os.remove('tmp.jpg')
        return output



class HomographyDataset(utils.data.Dataset):
    def __init__(self, path, rho, patch_size, noise):
        self.rho=rho
        self.patch_size=patch_size
        self.path = path
        self.image_path = []
        self.noise = noise

        random.seed(30)
        
        files = os.listdir(path)
        for file in files:
            self.image_path.append(file)
        

    def __getitem__(self,index):
        img = cv2.imread(self.path+'/'+str(self.image_path[index]),0)
        img = cv2.resize(img,(320,240))

        if self.noise == 'Vanilla':
            img = img
        elif self.noise == 'Blur5':
            img = cv2.blur(img,(5,5))
        elif self.noise == 'Blur10':
            img = cv2.blur(img,(10,10))
        elif self.noise == 'S&P':
            img = noisy('s&p', img)
        elif self.noise == 'Gaussian':
            img = noisy('gauss', img)
        elif self.noise == 'Compression':
            img = noisy('compress', self.path+'/'+str(self.image_path[index]))

        test_image = img.copy()

        top_x = random.randint(0+self.rho, 320-self.rho-self.patch_size)
        top_y = random.randint(0+self.rho, 240-self.rho-self.patch_size)
        
        top_left_point = (top_x, top_y)
        top_right_point = (top_x+self.patch_size, top_y)
        bottom_left_point = (top_x, top_y+self.patch_size)
        bottom_right_point = (top_x+self.patch_size, top_y+self.patch_size)

        four_points = [top_left_point, top_right_point, bottom_left_point, bottom_right_point]
        
        perturbed_four_points = []
        for point in four_points:
            perturbed_four_points.append( (point[0] + random.randint(-self.rho,self.rho), point[1] + random.randint(-self.rho,self.rho)) )

        H = cv2.getPerspectiveTransform( np.float32(four_points), np.float32(perturbed_four_points) )
        H_inverse = inv(H)

        warped_image = cv2.warpPerspective(img, H_inverse, (320,240))

        Ip1 = test_image[top_left_point[1]:bottom_right_point[1],top_left_point[0]:bottom_right_point[0]]

        Ip2 = warped_image[top_left_point[1]:bottom_right_point[1],top_left_point[0]:bottom_right_point[0]]

        training_image = np.dstack((Ip1, Ip2))
        H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))

        datum = (training_image, H_four_points)

        del img
        return datum


    def __len__(self):
        return len(self.image_path)
    