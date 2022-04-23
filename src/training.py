import torch
from tqdm import tqdm
from Model import ConvNet
from dataloader import train_dataloader, test_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 1e-3
num_epochs = 50


conv_net = ConvNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(conv_net.parameters(), lr=learning_rate)

conv_net.train()

for epoch in tqdm(range(num_epochs)):
    for i, (images, labels) in enumerate(train_dataloader):
        images.to(device)
        labels = torch.tensor(labels).to(device)
        optimizer.zero_grad()
        out = conv_net(images)
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch {epoch+1}, loss = {loss.item():.5f}")

with torch.no_grad():
    num_correct = 0
    n_samples = 0
    for images, labels in test_dataloader:
        images.to(device)
        labels = torch.tensor(labels).to(device)
        out = conv_net(images)

        # get value and index of max elem
        _, predictions = torch.max(out, 1)
        n_samples += labels.size(0)
        num_correct += (predictions == labels).sum().item()

    acc = num_correct/n_samples
    print(f'Accuracy: {acc}')

