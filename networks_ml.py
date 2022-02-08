import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing

from numba import jit
from torch import nn
from multiprocessing import Pool, cpu_count

rng = np.random.default_rng()


def get_dict(s: list) -> dict:
    out = {j: [] for j in set(x[0] for x in s)}
    for x in s:
        out[x[0]].append((x[1], x[2]))
    return out


@jit
def get_min_distances(graph: np.ndarray) -> np.ndarray:
    """Figures out minimum distance between all pairs in a network
    graph must be ndarray of type int
    
    Algorithm description:
    Initialize distance matrix to all zeros (meaning no connections found yet)
    For each network member, record direct neighbors as 1 in distance matrix
    For n = 1, 2, 3, ...
        For each network member, look at direct neighbors of those marked as n in previous step:
            If those neighbors are already != 0, do nothing
            Else record as n+1
        If distance matrix is all filled in, stop
    """
    size = graph.shape[0]
    neighbors = [[j for j in range(size) if graph[i, j]] for i in range(size)]
    distances = graph.copy()
    n = 1
    keep_going = True
    while keep_going:
        keep_going = False
        for i in range(size):
            for j in range(size):
                if i == j or distances[i, j] != n:
                    continue
                keep_going = True
                for k in neighbors[j]:
                    if i != k and distances[i, k] == 0:
                        distances[i, k] = n + 1
        n += 1
    for i in range(size):
        for j in range(size):
            if i != j and distances[i, j] == 0:
                distances[i, j] = -1
    return distances


class Network:

    def __init__(self, graph: np.ndarray, value_mean: float = 0.0, value_sd: float = 1.0, signal_life: int = 5, signal_error_sd: float = 0.1):
        self.graph = graph
        assert(graph.shape[0] == graph.shape[1])
        self.graph_size = graph.shape[0]
        self.values = rng.normal(loc=value_mean, scale=value_sd, size=self.graph_size)
        self.signal_life = signal_life
        self.signal_error_sd = signal_error_sd
        self._set_signals()
        self._network_distances = None
    
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

    def get_network_distances(self):
        if self._network_distances is None:
            self._network_distances = get_min_distances(self.graph)
        return self._network_distances
    
    def closeness_centrality(self):
        if -1 in self.get_network_distances():
            print('Warning: attempted to calculate closeness centrality on unconnected graph')
            return np.zeros(self.graph_size)
        else:
            return self.get_network_distances().sum(axis=0) ** -1
    
    def get_eigenvector_centrality(self):
        eigvals, eigvecs = np.linalg.eig(self.graph + np.eye(self.graph_size, dtype=int))
        return eigvecs[:, np.abs(eigvals).argmax()]



def gen_random_graph(size: int, p: float = 0.5) -> np.ndarray:
    """Creates a symmetric matrix of size (size x size) where proba of [off-diagonal element == 1] is p
    Diagonal elements are set to 0 
    """
    random_mat = rng.random((size, size))
    return (random_mat + random_mat.T < 2 * p).astype(int) * (1 - np.eye(size, dtype=int))


# now set up neural nets and training algorithm

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)


class MLModel(nn.Module):

    def __init__(
        self,
        # graph params
        graph_size: int,
        connection_proba: float = 0.5,
        value_mean: float = 0.0,
        value_sd: float = 1.0,
        signal_life: int = 5,
        signal_error_sd: float = 0.1,
        # model params
        max_signals: int = 2,
        n_hidden: int = 4,
        hidden_size: int = 50
    ):
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

        self.connection_proba = connection_proba
        self.value_mean = value_mean
        self.value_sd = value_sd
        self.signal_life = signal_life
        self.signal_error_sd = signal_error_sd
    
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

    def _get_nets_features_targets(
        self, n: int, threads: int
    ) -> typing.Tuple[list, list, list]:
        output_list = []
        while len(output_list) < n:
            pool_size = min(threads, n - len(output_list))
            with Pool(pool_size) as pool:
                output_list += pool.map(
                    get_nets_features_targets, [self] * pool_size
                )
        return tuple(zip(*output_list))

    def train(
        self,
        epochs: int = 100, lr: float = 1e-3,
        epoch_size: int = 20, threads: int = min(20, cpu_count())
    ) -> list:
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        loss_fn = nn.MSELoss()
        loss_history = []
        for i in range(epochs):
            _, features, targets = self._get_nets_features_targets(epoch_size, threads)
                    
            features = torch.concat(features)
            targets = torch.concat(targets)
            optimizer.zero_grad()
            loss = loss_fn(self.forward(features), targets)
            loss_history.append(loss.item())
            loss.backward()
            optimizer.step()
            if ((epochs >= 10 and (i+1) % (epochs // 10) == 0) or (i + 1 == epochs)):
                print(f'Epoch {i+1} of {epochs}: Loss = {loss:.3e}')
        return loss_history
    
    def _eval_single(self, net: Network, features: torch.Tensor, targets: torch.Tensor) -> pd.DataFrame:
        with torch.no_grad():
            scores = ((self.forward(features) - targets)**2).numpy()
        distances = net.get_network_distances()
        centralities = net.get_eigenvector_centrality()
        return pd.DataFrame(
            np.concatenate(
                (
                    np.stack((np.arange(net.graph_size, dtype=int), centralities), axis=1),
                    distances,
                    scores
                ),
                axis=1
            )
        )
    
    def eval(self, n: int, threads: int = min(20, cpu_count())) -> pd.DataFrame:
        nets, features, targets = self._get_nets_features_targets(n, threads)
        df = pd.concat((self._eval_single(n, f, t) for n, f, t in zip(nets, features, targets)))
        df.columns = (
            ['idx', 'centrality']
            + [f'd{i}' for i in range(self.graph_size)]
            + [f'score{i}' for i in range(self.graph_size)]
        )
        df['net'] = sum(([f'{i}'] * self.graph_size for i in range(n)), [])
        df.set_index(['net', 'idx'], drop = True, inplace = True)
        return df


def get_nets_features_targets(
    model: MLModel
) -> typing.Tuple[Network, torch.Tensor, torch.Tensor]:
    graph = gen_random_graph(model.graph_size, model.connection_proba)
    net = Network(graph, model.value_mean, model.value_sd, model.signal_life, model.signal_error_sd)
    features = model.get_features(net)
    targets = torch.tile(torch.from_numpy(net.values), (model.graph_size, 1)).float()
    return net, features, targets


def main():
    model = MLModel(20, connection_proba=0.2, signal_life=6, max_signals=3, hidden_size=100)

    loss_history = model.train(lr=1e-3, epochs=200)

    plt.plot(loss_history, linestyle='', marker='.', alpha=0.6)
    plt.show()

    model.eval(200).to_csv('test.csv')


if __name__ == '__main__':
    main()