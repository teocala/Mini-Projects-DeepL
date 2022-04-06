import torch 
from torch import nn

### For mini-project 1
class Model(nn.Module):
    def __init__(self) -> None:
        ## instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 3, kernel_size = 5,  stride = 1)
        self.convT1 = nn.ConvTranspose2d(3, 3, kernel_size=5, stride=1) 


        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        

    def load_pretrained_model(self ) -> None:
        ## This loads the parameters saved in bestmodel . pth into the model
        pass

    def train(self, train_input, train_target) -> None:
        # : train˙input : tensor of size (N , C , H , W ) containing a noisy version of the images.
        # : train˙target : tensor of size (N , C , H , W ) containing another noisy version of the
        # same images , which only differs from the input by their noise.
        batch_size = 100
        epochs = 5
        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs-1}')
            for batch_input, batch_target in zip(train_input.split(batch_size), train_target.split(batch_size)):
                output = self.predict(batch_input)
                loss = self.criterion(output, batch_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
    
    def predict(self, test_input) -> torch.Tensor:
        # : test˙input : tensor of size ( N1 , C , H , W ) that has to be denoised by the trained
        # or the loaded network .
        # : returns a tensor of the size ( N1 , C , H , W )
        
        # like the forward method
        
        y = self.conv1(test_input)
        y = self.convT1(y)
        
        return y
