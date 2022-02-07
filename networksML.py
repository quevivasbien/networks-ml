import torch
import numpy as np
import matplotlib.pyplot as plt
import typing

# from numba import jit
from torch import nn
# from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count

rng = np.random.default_rng()


def get_dict(s: list) -> dict:
    out = {j: [] for j in set(x[0] for x in s)}
    for x in s:
        out[x[0]].append((x[1], x[2]))
    return out


class Network:

    def __init__(self, graph: np.ndarray, value_mean: float = 0.0, value_sd: float = 1.0, signal_life: int = 5, signal_error_sd: float = 0.1):
        self.graph = graph
        assert(graph.shape[0] == graph.shape[1])
        self.graph_size = graph.shape[0]
        self.values = rng.normal(loc=value_mean, scale=value_sd, size=self.graph_size)
        self.signal_life = signal_life
        self.signal_error_sd = signal_error_sd
        self._set_signals()
    
    def _set_signals(self):
        signals = [[] for _ in range(self.graph_size)]
        # format of signal is (origin, direct source, value)
        to_send = [[(i, i, self.values[i])] for i in range(self.graph_size)]
        # keep track of where signals are coming from
        # only first signal from a given source about a given member will be kept
        unique_sources = [[(i, i)] for i in range(self.graph_size)]
        for _ in range(self.signal_life):
            received = [[] for _ in range(self.graph_size)]
            for i, to_send_ in enumerate(to_send):
                for j in range(self.graph_size):
                    if i != j and self.graph[i, j]:
                        warped_signals = [
                            (s[0], i, s[2] + rng.normal(scale=self.signal_error_sd))
                            for s in to_send_ if s[0] != j and (s[0], i) not in unique_sources[j]
                        ]
                        received[j] += warped_signals
                        unique_sources[j] += [(s[0], s[1]) for s in warped_signals]
            for sig, sent in zip(signals, to_send):
                sig += sent
            to_send = received
        # format as dict
        self.signals = [get_dict(s) for s in signals]


def gen_random_graph(size: int, p: float = 0.5) -> np.ndarray:
    """Creates a symmetric matrix of size (size x size) where proba of [off-diagonal element == True] is p"""
    random_mat = rng.random((size, size))
    return (random_mat + random_mat.T < 2 * p).astype(bool) * ~np.eye(size, dtype=bool)


# now set up neural nets and training algorithm

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)


class MLModel(nn.Module):

    def __init__(self, graph_size: int, max_signals: int, n_hidden: int = 4, hidden_size: int = 50):
        super().__init__()
        
        self.graph_size = graph_size
        self.max_signals = max_signals
        
        # input features are:
        # graph_size**2 features to describe the graph
        # graph_size to indicate guesser id
        # for each graph member * for each of max_signals:
        # graph_size features to indicate signal sender id
        # 1 for signal value
        self.n_other_features = graph_size * self.max_signals * (graph_size + 1)
        self.n_features = self.graph_size**2 + self.graph_size + self.n_other_features
        
        self.first = nn.Linear(self.n_features, hidden_size)
        self.hidden = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(n_hidden)
        ])
        self.last = nn.Linear(hidden_size, graph_size)

        self.apply(init_weights)
    
    def get_features(self, net: Network) -> torch.Tensor:
        assert(net.graph_size == self.graph_size)
        graph_features = torch.tile(
            torch.from_numpy(net.graph.flatten()), (self.graph_size,1)
        )
        id_features = torch.eye(self.graph_size)
        other_features = torch.zeros((self.graph_size, self.n_other_features))
        # iterate through signal receivers
        for i, signals in enumerate(net.signals):
            # iterate through signal subjects
            for j, s in signals.items():
                start_idx = self.max_signals * (self.graph_size + 1) * j
                # iterate_through signal senders
                for idx, (sender, value) in enumerate(s):
                    if idx >= self.max_signals:
                        break
                    offset = start_idx + (self.graph_size + 1) * idx
                    other_features[i, offset + sender] = 1.0
                    other_features[i, offset + self.graph_size] = value
        return torch.concat((graph_features, id_features, other_features), axis=-1).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.first(x))
        for h in self.hidden:
            x = x + torch.relu(h(x))
        return self.last(x)

    def train(
        self,
        connection_proba: float = 0.5,
        value_mean: float = 0.0, value_sd: float = 1.0, signal_life: int = 5, signal_error_sd: float = 0.1,
        epochs: int = 100, lr: float = 1e-3,
        epoch_size: int = 20, threads: int = min(20, cpu_count())
    ) -> list:
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        loss_fn = nn.MSELoss()
        loss_history = []
        for i in range(epochs):
            feature_target_list = []
            while len(feature_target_list) < epoch_size:
                pool_size = min(threads, epoch_size - len(feature_target_list))
                with Pool(pool_size) as pool:
                    feature_target_list += pool.map(
                        _get_features,
                        [(self, connection_proba, value_mean, value_sd, signal_life, signal_error_sd)] * pool_size
                    )
                    
            features = torch.concat(tuple(f[0] for f in feature_target_list))
            targets = torch.concat(tuple(f[1] for f in feature_target_list))
            optimizer.zero_grad()
            loss = loss_fn(self.forward(features), targets)
            loss_history.append(loss.item())
            loss.backward()
            optimizer.step()
            if ((i+1) % (epochs // 10) == 0):
                print(f'Epoch {i+1} of {epochs}: Loss = {loss:.3e}')
        return loss_history


def get_features(
    model: MLModel,
    connection_proba: float,
    value_mean: float, value_sd: float, signal_life: int, signal_error_sd: float
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    graph = gen_random_graph(model.graph_size, connection_proba)
    net = Network(graph, value_mean, value_sd, signal_life, signal_error_sd)
    features = model.get_features(net)
    targets = torch.tile(torch.from_numpy(net.values), (model.graph_size, 1)).float()
    return features, targets

def _get_features(
    args: typing.Tuple[MLModel, float, float, float, int, float]
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    return get_features(*args)


def main():
    """Currently just a test to run to make sure things are working well"""
    model = MLModel(12, 2)

    loss_history = model.train(connection_proba=0.2, lr=1e-2, epochs=100)

    plt.plot(loss_history, linestyle='', marker='.', alpha=0.6)
    plt.show()

    graph = gen_random_graph(model.graph_size, 0.5)
    net = Network(graph)

    with torch.no_grad():
        features = model.get_features(net)
        preds = model.forward(features)

    print(graph)
    print(net.values)
    print(preds.numpy())


if __name__ == '__main__':
    main()
