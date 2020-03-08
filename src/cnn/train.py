import torch


def train(args, train_loader, valid_loader, model, device, optimizer, criterion, logging):
    model.train()
    for epoch in range(args.epochs):
        logging.info(f'Epoch {epoch+1}')

        # train step (full epoch)
        running_loss_epoch = 0.0
        running_loss_batch = 0.0
        for idx, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss_batch += loss.item()
            if idx % 10 == 0 and idx != 0:
                logging.info(f'10 batches loss: {running_loss_batch/10}')
                running_loss_epoch += running_loss_batch
                running_loss_batch = 0.0

        logging.info(f'Train loss at the end of the epoch: {running_loss_epoch}')

        # valid step
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct/total
        logging.info(f'Validation accuracy: {accuracy}')
        # TODO F1 score, data augmentation, early stop, checkpointing...

