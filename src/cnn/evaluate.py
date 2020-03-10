import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report


def prettify_eval(accuracy, correct, avg_loss, class_report, n_instances):
    return '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n{}\n'.format(
        avg_loss, correct, n_instances, accuracy, class_report)


def evaluate(data_loader, model, device):
    model.eval()
    avg_loss = 0
    correct = 0
    y_output = []
    y_ground_truth = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            avg_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_output.append(pred)
            y_ground_truth.append(correct)

        avg_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    class_report = classification_report(y_ground_truth, y_output)
    return accuracy, correct, avg_loss, class_report, len(data_loader.dataset)


if __name__ == '__main__':
    # TODO command line tool, given a checkpoint and a path to data, evaluate and print result
    pass
