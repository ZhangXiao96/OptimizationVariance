import torch
import torch.nn as nn
import copy
import numpy as np
from lib import utils
from collections import Counter


class ModelWrapper(object):

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self._softmax = nn.Softmax(dim=-1)
        self._var_optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)


    def w_gradients_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if type(self.criterion) == nn.MSELoss:
            outputs = self._softmax(outputs)
            one_hot_targets = torch.zeros(inputs.size(0), outputs.size(-1),
                                          device=self.device).scatter(1,targets.unsqueeze(-1), 1)
            loss = self.criterion(outputs, one_hot_targets)
        else:
            loss = self.criterion(outputs, targets)
        loss.backward()
        gradients = []
        for name, param in self.model.named_parameters():
            grad_tensor = param.grad.data.cpu().numpy()
            gradients.append(grad_tensor.ravel())
        self.optimizer.zero_grad()
        return np.concatenate(gradients, axis=0)


    def train_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        acc = correct / targets.size(0)
        return loss.item(), acc, correct

    def train_on_batch_with_gradients_recorded(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = self.criterion(logits, targets)
        loss.backward()

        gradients = []
        for name, param in self.model.named_parameters():
            grad_tensor = param.grad.data.cpu().numpy()
            gradients.append(grad_tensor.ravel())
        gradients = np.concatenate(gradients, axis=0)

        self.optimizer.step()
        _, predicted = logits.max(1)
        correct = predicted.eq(targets).sum().item()
        acc = correct / targets.size(0)
        return loss.item(), acc, correct, logits.detach().cpu().numpy(), gradients

    def eval_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        return loss.item(), correct

    def get_optimization_var(self, data_loader, max_num=1000):
        original_weights = copy.deepcopy(self.model.state_dict())
        original_optim = copy.deepcopy(self.optimizer.state_dict())
        logits = []
        for i, (inputs, targets) in enumerate(data_loader):
            self.train()
            self.train_on_batch(inputs, targets)
            self.eval()
            logit, _, _ = self.predict_all(data_loader, max_number=max_num)
            logits.append(logit)

            self.model.load_state_dict(original_weights)
            self.optimizer.load_state_dict(original_optim)

        logits = np.array(logits)
        ov = np.array([np.mean(np.var(logits[:, _, :], axis=0)) for _ in range(logits.shape[1])])
        return np.mean(ov / np.mean(logits ** 2, axis=(0, -1), keepdims=False))

    def predict_on_batch(self, inputs):
        inputs = inputs.to(self.device)
        logits = self.model(inputs)
        _, predicted = logits.max(1)
        return logits.data.cpu().numpy(), predicted.data.cpu().numpy()

    def eval_all(self, test_loader):
        test_loss = 0
        test_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                loss, correct = self.eval_on_batch(inputs, targets)
                total += targets.size(0)
                test_loss += loss
                test_correct += correct
            test_loss /= (batch_idx+1)
            test_acc = test_correct / total
        return test_loss, test_acc

    def predict_all(self, test_loader, max_number=None):
        with torch.no_grad():
            logits = []
            labels = []
            truth = []
            nb = 0
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                logit, label = self.predict_on_batch(inputs)
                logits.append(logit)
                labels.append(label)
                truth.append(targets.cpu().numpy())
                nb += len(label)
                if max_number:
                    if nb >= max_number:
                        break
            nb_all = min(max_number, nb) if max_number else nb
            logits = np.concatenate(logits, axis=0)[:nb_all]
            labels = np.concatenate(labels, axis=0)[:nb_all]
            truth = np.concatenate(truth, axis=0)[:nb_all]
        return logits, labels, truth

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()
