import torch
import numpy as np 
from torch import nn


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        self.embedding_layer = nn.Embedding(embeddings.shape[0],embeddings.shape[1])  # EX4
        self.embeddings_size = embeddings.shape[1]

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embedding_layer.load_state_dict({'weight': torch.from_numpy(embeddings)}) # EX4

        # 3 - define if the embedding layer will be frozen or finetuned
        self.embedding_layer.weight.requires_grad = trainable_emb  # EX4

        # 4 - define a non-linear transformation of the representations
        hidden_size = 32
        self.linear = nn.Linear(embeddings.shape[1], hidden_size)
        self.func = nn.ReLU()  # EX5

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.output = nn.Linear(hidden_size, output_size)  # EX5

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        # 1 - embed the words, using the embedding layer
        batch_size = len(x)
 
        embeddings = self.embedding_layer(x)  # EX6

        # 2 - construct a sentence representation out of the word embeddings
        representations = torch.zeros([batch_size, self.embeddings_size]).to('cuda')
        for i in range(batch_size):
            representations_mean = torch.sum(embeddings[i], dim=0) / lengths[i]  # EX6
            representations_max = torch.max(embeddings[i], dim=0)
            representations[i] = torch.cat((representations_mean, representations_max), 1)

        # 3 - transform the representations to new ones.

        representations = self.func(self.linear(representations))  # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.output(representations)  # EX6

        return logits


class LSTM(nn.Module):
    def __init__(self, output_size, embeddings, trainable_emb=False, bidirectional=False):

        super(LSTM, self).__init__()
        self.hidden_size = 100
        self.num_layers = 1
        self.bidirectional = bidirectional

        self.representation_size = 2 * \
            self.hidden_size if self.bidirectional else self.hidden_size

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        self.lstm = nn.LSTM(dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)

        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True)

        self.linear = nn.Linear(self.representation_size, output_size)

    def last_timestep(self, outputs, lengths, bidirectional=False):
  
        #Returns the last output of the LSTM taking into account the zero padding and 
        #the bidirectional compoment
   
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            if last_forward.shape != last_backward.shape:
                return torch.cat((last_forward.unsqueeze(0), last_backward), dim=-1)
            else:
                return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    #Define function to get the directions of forward and backward for sequence in LSTM
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1).to(torch.int64)
        return outputs.gather(1, idx).squeeze()   

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x)
        X = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)

        ht, _ = self.lstm(X)

        # ht is batch_size x max(lengths) x hidden_dim
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht,batch_first=True)

        # pick the output of the lstm corresponding to the last word
        # TODO: Main-Lab-Q2 (Hint: take actual lengths into consideration)
        representations = self.last_timestep(ht,lengths,self.bidirectional)
   

        logits = self.linear(representations)

        return logits