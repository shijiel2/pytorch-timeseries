from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.certify import Smooth

from typing import cast, Any, Dict, List, Tuple, Optional


class BaseTrainer:
    """Trains an inception model. Dataset-specific trainers should extend this class
    and implement __init__, get_loaders and save functions.
    See UCRTrainer in .ucr.py for an example.

    Attributes
    ----------
    The following need to be added by the initializer:
    model:
        The initialized inception model
    data_folder:
        A path to the data folder - get_loaders should look here for the data
    model_dir:
        A path to where the model and its predictions should be saved

    The following don't:
    train_loss:
        The fit function fills this list in as the model trains. Useful for plotting
    val_loss:
        The fit function fills this list in as the model trains. Useful for plotting
    test_results:
        The evaluate function fills this in, evaluating the model on the test data
    """
    model: nn.Module
    data_folder: Path
    model_dir: Path
    train_loss: List[float] = []
    val_loss: List[float] = []
    test_results: Dict[str, float] = {}
    input_args: Dict[str, Any] = {}

    def fit(self, batch_size: int = 64, num_epochs: int = 100,
            val_size: float = 0.2, learning_rate: float = 0.01,
            patience: int = 10, device='cuda') -> None:
        """Trains the inception model

        Arguments
        ----------
        batch_size:
            Batch size to use for training and validation
        num_epochs:
            Maximum number of epochs to train for
        val_size:
            Fraction of training set to use for validation
        learning_rate:
            Learning rate to use with Adam optimizer
        patience:
            Maximum number of epochs to wait without improvement before
            early stopping
        """
        train_loader, val_loader = self.get_loaders(batch_size, mode='train', val_size=val_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_val_loss = np.inf
        patience_counter = 0
        best_state_dict = None

        self.model.train()
        for epoch in range(num_epochs):
            epoch_train_loss = []
            for x_t, y_t in train_loader:
                x_t, y_t = x_t.to(device), y_t.to(device)
                optimizer.zero_grad()
                output = self.model(x_t)
                if len(y_t.shape) == 1:
                    train_loss = F.binary_cross_entropy_with_logits(
                        output, y_t.unsqueeze(-1).float(), reduction='mean'
                    )
                else:
                    train_loss = F.cross_entropy(output, y_t.argmax(dim=-1), reduction='mean')

                epoch_train_loss.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
            self.train_loss.append(np.mean(epoch_train_loss))

            epoch_val_loss = []
            self.model.eval()
            for x_v, y_v in cast(DataLoader, val_loader):
                with torch.no_grad():
                    x_v, y_v = x_v.to(device), y_v.to(device)
                    output = self.model(x_v)
                    if len(y_v.shape) == 1:
                        val_loss = F.binary_cross_entropy_with_logits(
                            output, y_v.unsqueeze(-1).float(), reduction='mean'
                        ).item()
                    else:
                        val_loss = F.cross_entropy(output,
                                                   y_v.argmax(dim=-1), reduction='mean').item()
                    epoch_val_loss.append(val_loss)
            self.val_loss.append(np.mean(epoch_val_loss))

            print(f'Epoch: {epoch + 1}, '
                  f'Train loss: {round(self.train_loss[-1], 3)}, '
                  f'Val loss: {round(self.val_loss[-1], 3)}')

            if self.val_loss[-1] < best_val_loss:
                best_val_loss = self.val_loss[-1]
                best_state_dict = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter == patience:
                    if best_state_dict is not None:
                        self.model.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))
                    print('Early stopping!')
                    return None
    
    def predict(self, x, num_classes):

        self.model.eval()

        with torch.no_grad():
            preds = self.model(x)
            if num_classes == 2:
                preds = torch.sigmoid(preds)
            else:
                preds = torch.softmax(preds, dim=-1)
            preds = preds.detach().numpy()

        if num_classes > 2:
            return np.argmax(preds, axis=-1)
        else:
            return (preds > 0.5).astype(int)
        
    def lb_keogh_envelope(self, x, w):
        """
        Compute the LB_Keogh envelope for a time series tensor x of shape [channels, length]
        without adding a batch dimension.

        Args:
            x (torch.Tensor): Input tensor of shape [channels, length].
            w (int): Half-window size. The total window size will be 2*w + 1.

        Returns:
            U (torch.Tensor): Upper envelope of shape [channels, length].
            L (torch.Tensor): Lower envelope of shape [channels, length].
        """
        kernel_size = 2 * w + 1

        # Pad the time series along the length dimension using replicate padding.
        # The padding is applied on both sides.
        x_padded = F.pad(x, (w, w), mode='replicate')  # shape: [channels, length + 2*w]

        # Create sliding windows along the length dimension.
        # The resulting tensor has shape [channels, length, kernel_size].
        x_windows = x_padded.unfold(dimension=1, size=kernel_size, step=1)

        # Compute the upper envelope (max over the window) and the lower envelope (min over the window)
        U = x_windows.max(dim=-1)[0]
        L = x_windows.min(dim=-1)[0]

        return U, L
    
    def lb_keogh_consume(self, x, w):
        U, L = self.lb_keogh_envelope(x, w)
        diff = torch.max(U - x, x - L)
        return diff.sum()

        
    def certify(self, rm_batch_size: int=1000, sigma: float=0.2, device='cuda') -> None:
        num_classes = self.get_num_classes()
        smoothed_model = Smooth(self.model, num_classes=num_classes, sigma=sigma)

        test_loader, _ = self.get_loaders(batch_size=1, mode='test')

        true_list, preds_list = [], []
        for x, y in test_loader:
            with torch.no_grad():
                x, y = x.to(device).squeeze(0), y.to(device).squeeze(0)
                true_list.append(y.cpu().detach().numpy())
                pred, rad = smoothed_model.rm_certify(x, n0=100, n=10000, alpha=0.001, batch_size=rm_batch_size)
                lb_consume = self.lb_keogh_consume(x, 3)
                print(f'Prediction: {pred}, Radius: {rad}, LB consume: {lb_consume}')
                


    def evaluate(self, batch_size: int = 64, device='cuda') -> None:

        test_loader, _ = self.get_loaders(batch_size, mode='test')

        self.model.eval()

        true_list, preds_list = [], []
        for x, y in test_loader:
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                true_list.append(y.cpu().detach().numpy())
                preds = self.model(x)
                if len(y.shape) == 1:
                    preds = torch.sigmoid(preds)
                else:
                    preds = torch.softmax(preds, dim=-1)
                preds_list.append(preds.cpu().detach().numpy())

        true_np, preds_np = np.concatenate(true_list), np.concatenate(preds_list)

        self.test_results['roc_auc_score'] = roc_auc_score(true_np, preds_np)
        print(f'ROC AUC score: {round(self.test_results["roc_auc_score"], 3)}')

        self.test_results['accuracy_score'] = accuracy_score(
            *self._to_1d_binary(true_np, preds_np)
        )
        print(f'Accuracy score: {round(self.test_results["accuracy_score"], 3)}')

    @staticmethod
    def _to_1d_binary(y_true: np.ndarray, y_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(y_true.shape) > 1:
            return np.argmax(y_true, axis=-1), np.argmax(y_preds, axis=-1)

        else:
            return y_true, (y_preds > 0.5).astype(int)

    def get_num_classes(self) -> int:
        test_loader, _ = self.get_loaders(64, mode='test')
        
        # Retrieve the first batch from the data_loader
        for _, y in test_loader:
            with torch.no_grad():
                y.detach().numpy()
                break

        # If y has more than one dimension, assume the last dimension represents classes.
        if len(y.shape) > 1:
            return y.shape[-1]
        else:
            # Otherwise, it is used in a binary classification problem
            return 2

    
    def get_loaders(self, batch_size: int, mode: str,
                    val_size: Optional[float] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Return dataloaders of the training / test data

        Arguments
        ----------
        batch_size:
            The batch size each iteration of the dataloader should return
        mode: {'train', 'test'}
            If 'train', this function should return (train_loader, val_loader)
            If 'test', it should return (test_loader, None)
        val_size:
            If mode == 'train', the fraction of training data to use for validation
            Ignored if mode == 'test'

        Returns
        ----------
        Tuple of (train_loader, val_loader) if mode == 'train'
        Tuple of (test_loader, None) if mode == 'test'
        """
        raise NotImplementedError

    def save_model(self, savepath: Optional[Path] = None) -> Path:
        raise NotImplementedError
