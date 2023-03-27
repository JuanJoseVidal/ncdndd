import torch
import numpy as np

def train(model, loss_func, data_loader, weights=None, epochs=5000, patience=100):
    opt = torch.optim.Adam(model.parameters(), amsgrad=True)
    best_loss_cte = 1000
    patience_counter = 0
    loss_history = []
    for epoch in range(epochs):
        loss_epoch = []
        for x_mini, y_mini in data_loader:
            opt.zero_grad()
            y_predicted = model(x_mini)
            loss, loss_cte = loss_func(y_predicted, y_mini, weights=weights)
            loss.backward()
            opt.step()
            loss_epoch.append(loss.detach().numpy())
        loss_history.append(np.mean(loss_epoch))
        if loss_cte < best_loss_cte:
            best_loss_cte = loss_cte
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == patience:
                print(f"Overfitting mudafuka. Final epoch {epoch}.")
                break
    return model, loss_history

