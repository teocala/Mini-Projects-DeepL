import torch 
from torch import nn
from .others.utilities import *
from pathlib import Path

### For mini-project 1
class Model(nn.Module):
    def __init__(self) -> None:
        ## instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()


        self.unet = UNet(3,3)
        #self.unet = ResUNet(3,3)

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00085)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,[5,7],gamma=0.1)


    def load_pretrained_model(self ) -> None:
        ## This loads the parameters saved in bestmodel . pth into the model
        model_path = Path(__file__).parent / "bestmodel.pth"
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint)


    def train(self, train_input, train_target, num_epochs) -> None:
        # : train˙input : tensor of size (N , C , H , W ) containing a noisy version of the images.
        # : train˙target : tensor of size (N , C , H , W ) containing another noisy version of the
        # same images , which only differs from the input by their noise.

        batch_size = 20

        print('Training the model...')

        # Using a GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using {} as device".format(device))

        # Convert the data into float type
        train_input = train_input.float()
        train_target = train_target.float()

        # Move to the device
        train_input = train_input.to(device)
        train_target = train_target.to(device)
        self.unet = self.unet.to(device)


        print("Doing data augmentation")
        trainset = MyData(train_input,train_target)
        trainloader = DataLoader(trainset,batch_size,shuffle=True,num_workers=0,collate_fn=collate_batch)


        for epoch in range(num_epochs):
            total_loss = 0
            for batch_input, batch_target in trainloader:
                output = self.predict(batch_input)
                loss = self.criterion(output, batch_target)
                total_loss += loss * batch_size * 3
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()


            total_loss /= 3*train_input.size(0)

            print(f'Epoch {epoch}/{num_epochs - 1} Training Loss {total_loss}  PSNR {error} DB' )


    def predict(self, test_input) -> torch.Tensor:
        # : test˙input : tensor of size ( N1 , C , H , W ) that has to be denoised by the trained
        # or the loaded network .
        # : returns a tensor of the size ( N1 , C , H , W )
        
        # like the forward method

        x = self.unet(test_input.float())
        x = torch.clamp(x,0,255)


        return x
