import torch
import torchvision
from torch.optim import lr_scheduler
from tqdm import tqdm
from resnet_dataloader import train_dataloader, test_dataloader
from pretrained_resnet_with_extra_layers import resnet_and_fc
from torch.utils.tensorboard import SummaryWriter


board = SummaryWriter('tensorboard_res/raport')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LEARNING_RATE = 1e-3
NUM_EPOCHS = 20


resnet_model = resnet_and_fc.to(device)

# visualize model in tensorboard
batch_images = next(iter(train_dataloader))[0]
board.add_graph(resnet_model, next(iter(train_dataloader))[0])
batch_grid = torchvision.utils.make_grid(batch_images)
board.add_image('Batch of images', batch_grid)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet_model.parameters(), lr=LEARNING_RATE)
lr_step_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=17, gamma=0.1)


resnet_model.train()

current_loss = 0
current_pred = 0

for epoch in tqdm(range(NUM_EPOCHS)):
    for i, (images, labels) in enumerate(train_dataloader):
        images.to(device)
        labels = torch.tensor(labels.to(device))
        optimizer.zero_grad()
        out = resnet_model(images.to(device)).to(device)
        loss = criterion(out, labels.to(device))

        loss.backward()
        optimizer.step()

        current_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        current_pred += (predicted == labels).sum().item()

        if (i+1) % 1 == 0:
            print(f"Epoch {epoch+1}, loss = {loss.item():.5f}")

        # Plot accuracy and loss in tensorboard
        if (i + 1) % 10 == 0:
            board.add_scalar('training loss', current_loss / 10, epoch * len(train_dataloader) + i)
            board.add_scalar('training accuracy', current_pred / 10, epoch * len(train_dataloader) + i)
            current_loss = 0
            current_pred = 0

        # change lr
    lr_step_scheduler.step()

torch.save(resnet_model.state_dict(), 'model_resnet.pth')
board.close()

# test loop
with torch.no_grad():
    num_correct = 0
    n_samples = 0
    for images, labels in test_dataloader:
        images.to(device)
        labels = torch.tensor(labels.to(device)).to(device)
        out = resnet_model(images.to(device)).to(device)

        # get value and index of max elem
        _, predictions = torch.max(out, 1)
        n_samples += labels.size(0)
        # sum up the correct responses of the model
        num_correct += (predictions == labels).sum().item()

    acc = num_correct/n_samples
    print(f'Test accuracy: {acc}')
