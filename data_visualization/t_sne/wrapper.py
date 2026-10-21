import random
import torch
import numpy as np
from torch.autograd import Variable

def chunks(n, pij, i, j):
    """Yield successive n-sized chunks from pij, i and j."""
    endpoints = []
    start = 0

    for stop in range(n, len(pij) + n, n):
        stop = min(stop, len(pij))
        endpoints.append((start, stop))
        start = stop

    random.shuffle(endpoints)

    for start, stop in endpoints:
        yield pij[start:stop], i[start:stop], j[start:stop]


class Wrapper:
    def __init__(self, model, cuda=True, epochs=1000, batchsize=1024):
        self.batchsize = batchsize
        self.epochs = epochs
        self.cuda = cuda
        self.model = model
        if cuda:
            self.model.cuda()
        # TODO: добавьте оптимизатор
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-3)

    def fit(self, pij, i, j, offset=0, *args):
        self.model.train()
        if self.cuda:
            self.model.cuda()
        pij = np.asarray(pij, dtype=np.float32)
        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)

        if not (len(pij) == len(i) == len(j)):
            raise ValueError(
                "pij, i and j must have the same length"
            )
        for epoch in range(self.epochs):
            total = 0.0
            for itr, datas in enumerate(chunks(self.batchsize, pij, i, j)):
                datas = [Variable(torch.from_numpy(data)) for data in datas]
                if self.cuda:
                    datas = [data.cuda() for data in datas]
                    pij_batch, i_batch, j_batch = datas
                    # TODO: вычислите логиты модели и градиенты от datas
                    # datas = [pij, i, j]
                    # pij - значения сходства между точками данных
                    # i, j - индексы точек
                    self.optimizer.zero_grad()
                    loss = self.model(*datas)
                    loss.backward()
                    self.optimizer.step()
                    total += loss.item()
                    
            msg = "Train Epoch: {} \tLoss: {:.6e}"
            msg = msg.format(epoch + offset, total / (len(args[0]) * 1.0))
            print(msg)
