from dataset import HomographyDataset
from model import DeepHomographyModel
from torchsummary import summary
import torch.nn as nn
import torch.utils as utils
import torch.optim as optim
import torch
import argparse
import time


def parseArgs():
    parser = argparse.ArgumentParser(description='Deep Homography Estimation in Pytorch')
    parser.add_argument('--epochs', help='number of epochs to train', default=50, type=int)
    parser.add_argument('--lr', help='learning rate', default='0.005', type=float)
    parser.add_argument('--momentum', default='0.9', type=float)
    parser.add_argument('--rho', default='32', type=int)
    parser.add_argument('--patch_size', default='128', type=int)
    parser.add_argument('--batch_size', help='batch size', default='64',type=int)
    parser.add_argument('--data_year', help='the version of MS-COCO dataset used. Default is "2014"', default="2014", type=str)
    parser.add_argument('--noise', help='the noise to use when generating data', default='Vanilla', choices=['Vanilla', 'Blur5', 'Blur10', 'Compression', "Gaussian", "S&P", "All"])
    args = parser.parse_args(args=[])
    return args


def train_model(args, model, criterion, optimizer, TrainLoader, ValLoader):
    loader = TrainLoader
    print("Starting Training")
    for epoch in range(args.epochs):
        testError = 0
        torch.cuda.empty_cache()
        for phase in ['train', 'val']:
            
            start_time = time.time()
            
            if phase == 'train':
                loader=TrainLoader
            else:
                loader=ValLoader
            
            testError = 0
            for index, (images, target) in enumerate(loader):
                optimizer.zero_grad()
                images = images.to(device)
                target = target.to(device)

                images = images.permute(0,3,1,2).float()
                target = target.float()
                
                outputs = model.forward(images)
                
                loss = criterion(outputs, target.view(-1,8))
                testError += loss.item()**0.5
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                del loss
                del outputs
                del images
                del target

            cornerAvgError = testError /len(loader)
            elapsed_time = time.time() - start_time
            print('**********************')
            print('Phase: '+str(phase))
            print('Epoch Number: [{}/{}] | Corner Avg Error (CAE): {:.6f}'.format(epoch+1, args.epochs, cornerAvgError))
            print('Time elapsed: '+str(elapsed_time)+' seconds')
                
    del TrainLoader
    del ValLoader

def test_model(args, model, criterion, TestLoader, noise):
    start_time = time.time()
    testError = 0
    nbatch = 0
    torch.cuda.empty_cache()

    model.eval()

    with torch.no_grad():
      start_time = time.time()         
      for index,(images, target) in enumerate(TestLoader):

          nbatch += 1
          images = images.to(device)
          target = target.to(device)
  
          images = images.permute(0,3,1,2).float()
          target = target.float()
  
          outputs = model.forward(images)
          testError += criterion(outputs, target.view(-1,8)).item()**0.5
        
    print("************************")                    
    print("Test Error")
    print("Type of noise: "+noise)
    cae = testError/nbatch
    print("Corner Avg Error: "+str(cae))
    elapsed_time = time.time() - start_time
    print("Time elapsed: "+str(elapsed_time))
    print("************************")


def main():
    args = parseArgs()

    model = DeepHomographyModel().to(device)
    print("Arguments are:")
    print(args)
    print("------------------")
    

    if (args.noise=="All"):
        noises = ['Vanilla', 'Blur5', 'Blur10', 'Gaussian', 'S&P', 'Compression']
    else:
        noises = [args.noise]

    testNoises = ['Vanilla', 'Blur5', 'Blur10', 'Gaussian', 'S&P', 'Compression']
    
    for noise in noises:
        trainData = HomographyDataset("data/train"+str(args.data_year), args.rho, args.patch_size, noise)
        valData = HomographyDataset("data/val"+str(args.data_year), args.rho, args.patch_size, noise)

        TrainLoader = utils.data.DataLoader(trainData, args.batch_size)
        ValLoader = utils.data.DataLoader(valData, args.batch_size)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_model(args, model, criterion, optimizer, TrainLoader, ValLoader)
        torch.save(model.state_dict(), "model_final_"+str(noise))
        
        for testNoise in testNoises:
            testData = HomographyDataset("data/test"+str(args.data_year), args.rho, args.patch_size, testNoise)
            TestLoader = utils.data.DataLoader(testData, args.batch_size)
            test_model(args, model, criterion, TestLoader, testNoise)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main()