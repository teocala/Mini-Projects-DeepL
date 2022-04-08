import torch 
from torch import nn

### For mini-project 1
class Model(nn.Module):
    def __init__(self) -> None:
        ## instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 5, stride = 1),  # N , 32, 28 , 28
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=5,stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2), # N , 32 , 14 , 14
            nn.Conv2d(32,64, kernel_size = 5, stride = 1, padding = 2), # N , 64, 10, 10
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=5,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2 , stride = 2), # N , 64 , 5 , 5
            nn.Conv2d(64,128, kernel_size = 5, stride = 1) # N , 128 , 1 , 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,64, kernel_size = 5, stride = 1), # N, 64, 5, 5
            nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True), # N , 64, 10, 10
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,kernel_size=5,stride=1, padding = 2), # N, 64, 10, 10
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=5,stride=1), # N , 32, 14, 14
            nn.Upsample(scale_factor=2,mode = 'bilinear', align_corners=True), # N, 32, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(32,32,kernel_size=5,stride=1, padding=2), # N, 32, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(32,3,kernel_size=5,stride=1) # N, 3, 32, 32
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        

    def load_pretrained_model(self ) -> None:
        ## This loads the parameters saved in bestmodel . pth into the model
        pass

    def train(self, train_input, train_target) -> None:
        # : train˙input : tensor of size (N , C , H , W ) containing a noisy version of the images.
        # : train˙target : tensor of size (N , C , H , W ) containing another noisy version of the
        # same images , which only differs from the input by their noise.

        batch_size = 100
        epochs = 500
        total_loss = 0

        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs-1} Training Loss {total_loss}')
            total_loss = 0
            for batch_input, batch_target in zip(train_input.split(batch_size), train_target.split(batch_size)):
                output = self.predict(batch_input)
                loss = self.criterion(output, batch_target)
                total_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def predict(self, test_input) -> torch.Tensor:
        # : test˙input : tensor of size ( N1 , C , H , W ) that has to be denoised by the trained
        # or the loaded network .
        # : returns a tensor of the size ( N1 , C , H , W )
        
        # like the forward method
        
        y = self.encoder(test_input)
        y = self.decoder(y)
        
        return y
