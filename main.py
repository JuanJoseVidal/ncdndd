import torch
from nn.models import QModelCte, QModelCteNonCrossing, QModel, CteModel
from nn.train import train
from data.prepare_data import read_data, scale_data, data_pytorch
from aux_code.loss_functions import QuantileLossCTE, general_loss_function
from aux_code.aux_code import set_seed
from pandas import DataFrame

def main(quantiles=[.9], seeds=range(1), epochs_train=5000):
    df, response_name, weights_name = read_data()
    df_scaled, max_values, weights = scale_data(df, weights_name)
    dataloader, x, y = data_pytorch(df_scaled, response_name)

    lf_sum = []
    predictions_list = []
    loss_history_list = []
    for seed in seeds:
        set_seed(seed)

        model_cte = QModelCteNonCrossing(x.shape[1], len(quantiles))
        loss_func = QuantileLossCTE(quantiles)
        
        model, loss_history = train(model_cte, loss_func, dataloader, weights=weights, epochs=epochs_train)
        preds = model(torch.Tensor(x)).detach().numpy() * max_values[response_name]
        
        predictions_list.append(preds)
        lf_sum.append(
            [sum(general_loss_function(
                torch.tensor(preds[:, i]),
                torch.tensor(preds[:, i] + preds[:, i + len(quantiles)]),
                torch.tensor(y) * max_values[response_name], quantiles[i], 100)).detach().item()
                for i in range(len(quantiles))])
        loss_history_list.append(loss_history)

        print("Finished seed: ", seed)
        print("Loss: ", lf_sum[-1])

    for q in range(len(quantiles)):
        q_preds_df = DataFrame([[i[q] for i in p] for p in predictions_list])
        cte_preds_df = DataFrame([[i[q + len(quantiles)] for i in p] for p in predictions_list])

        q_preds_df.transpose().to_csv(f"./export/preds_q_{quantiles[q]}.csv")
        cte_preds_df.transpose().to_csv(f"./export/preds_cte_{quantiles[q]}.csv")


    DataFrame(loss_history_list).transpose().to_csv(f"./export/loss_history.csv")
    DataFrame(lf_sum, columns=quantiles).to_csv(f"./export/lf_sum.csv")


if __name__ == '__main__':
    import time
    startTime = time.time()

    main(quantiles=[.5, .6, .7, .8, .9, .925, .95, .975, .99], seeds=range(50), epochs_train=20000)

    executionTime = (time.time() - startTime)
    print('Execution time (seg): ' + str(executionTime))

