import numpy as np
import torch

import time
import matplotlib.pyplot as plt
import seaborn

from PyTorch_Transformer_model import create_model


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """
    Object for holding a batch of data with mask during training.

    Notes:
    ------
    shifting right works as follows:
    let Y = [y_1, y_2, ... y_n]^T be output sequence.
    Then input is constructed as Y^{in} = [y_1, ..., y_{n-1}]^T and output as Y^{out} = [y_2, ..., y_{n}]^T. 
    Training is performed such that transformer f is trying to map Y^{in}_i into Y^{out}_i 
    f(Y^{in}_i) -> Y^{out}_i and thanks to the shift it can be trained in parallel for every i.  
    """
    def __init__(self, src, trg = None, pad = 0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:] # shift right
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


# Training Loop:
def run_epoch(data_iter, model, loss_fun):
    """
    Training and logging
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_fun(out, batch.trg_y, batch.ntokens) # Batch loss
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print('Epoch step: {} Loss: {} Tokens per sec: {}'.format(i, loss/batch.ntokens, tokens/elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens




global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(torch.nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        true_dist.requires_grad_(False)
        return self.criterion(x, true_dist)





if __name__ == '__main__':
    # Learning rate warm start: ---------------------- 
    # if False:
    #   opts = [NoamOpt(512, 1, 4000, None), 
    #       NoamOpt(512, 1, 8000, None),
    #       NoamOpt(256, 1, 4000, None)]
    #   plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    #   plt.legend(["512:4000", "512:8000", "256:4000"])
    #   plt.show()

    # Example of label smoothing: ----------------------
    # if False:
    #   crit = LabelSmoothing(5, 1, 0.4)
    #   predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
    #                                [0, 0.2, 0.7, 0.1, 0], 
    #                                [0, 0.2, 0.7, 0.1, 0]])
    #   v = crit(predict.log(), 
    #            torch.LongTensor([2, 1, 0]))
    #   # Show the target distributions expected by the system.
    #   plt.imshow(crit.true_dist)
    #   plt.show()


    #######################
    ### Synthetic data: ###
    def data_gen(V, batch, nbatches):
        """
        Generate random data for a src-tgt copy task.
        """
        for i in range(nbatches):
            data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
            data[:, 0] = 1
            src = data.detach().clone().to(torch.int64)
            tgt = src.clone().to(torch.int64)
            yield Batch(src, tgt, 0)

    class SimpleLossCompute:
        """
        A simple loss compute and train function.
        """
        def __init__(self, generator, criterion, opt=None):
            self.generator = generator
            self.criterion = criterion
            self.opt = opt

        def __call__(self, x, y, norm):
            x = self.generator(x)
            loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                                  y.contiguous().view(-1)) / norm
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
            return loss.data.item() * norm

    # Train the simple copy task.
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = create_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model, 
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model, 
                        SimpleLossCompute(model.generator, criterion, None)))



    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len-1):
            out = model.decode(memory, src_mask, 
                               ys, 
                               subsequent_mask(ys.size(1)).type_as(src.data))
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, 
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        return ys

    model.eval()
    src = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]])
    src_mask = torch.ones(1, 1, 10)
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))