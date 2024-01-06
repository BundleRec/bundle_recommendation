import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class TsfCompletion(nn.Module):
    def __init__(self, 
                 item_num, 
                 factors=32,
                 epochs=20, 
                 lr=0.01, 
                 reg_1=0.001,
                 num_layers=2,
                 num_heads=4, 
                 item_embedding=None,
                 gpuid='0', 
                 vali_tra = None,
                 input_type='concat'):
        """
        Transformers bundle completion Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(TsfCompletion, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.epochs = epochs
        self.lr = lr
        self.train_mat = self.build_vali_train(vali_tra)
        self.input_type = input_type

        padding_embed = torch.zeros(1, factors)
        self.embed_item = nn.Embedding.from_pretrained(torch.cat((item_embedding, padding_embed), dim=0), freeze=False)
        self.transformer_layers = nn.TransformerEncoderLayer(d_model=factors, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layers, num_layers=num_layers)

        if self.input_type == 'concat':
            linear_input_dim = 10 * factors
        else:
            linear_input_dim = factors
        
        self.fc_layers = nn.Sequential(
            nn.Linear(linear_input_dim, 2*linear_input_dim),
            nn.ReLU(),
            nn.Linear(2*linear_input_dim, item_num)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=reg_1)

    def forward(self, item_ids):
        input_ids = self.embed_item(item_ids) # bach_size * seq_len * embed_dim
        encoder_input = input_ids
        encoder_input = encoder_input.permute(1, 0, 2)
        # transformer encoder
        if self.input_type == 'concat':
            hidden_state = self.transformer_encoder(encoder_input).view(encoder_input.shape[1], -1) # batch_size * embed_dim
            padding_size = 200 - hidden_state.shape[1]
            hidden_state = F.pad(hidden_state, (0, padding_size))
            # print(hidden_state.shape)
        elif self.input_type == 'mean':
            hidden_state = self.transformer_encoder(encoder_input).mean(dim=0) # batch_size * embed_dim
        # hidden_state = self.transformer_encoder(encoder_input).mean(dim=0) # batch_size * embed_dim
        # fc layers
        output = self.fc_layers(hidden_state)
        output = F.log_softmax(output, dim=-1)

        return output
    
    def fit(self, train_loader, val_loader=None):
        """
        Fit the model with train_loader
        Parameters
        ----------
        train_loader : DataLoader, training data iterator
        val_loader : DataLoader, validation data iterator
        """
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0
            for itemids, targets in tqdm(train_loader):
                # item_ids = itemids
                if torch.cuda.is_available():
                    item_ids = itemids.cuda()
                    targets = targets.cuda()
                else:
                    item_ids = itemids.cpu()
                    targets = targets.cpu()
                self.optimizer.zero_grad()
                output = self.forward(item_ids)
                loss = self.criterion(output, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            print("Epoch %d, train loss %.4f" % (epoch, train_loss))
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print("Epoch %d, val loss %.4f" % (epoch, val_loss))

    def predict(self, user_ids, candidate_items):
        """
        Predict the score of given user_ids and item_ids
        Parameters
        ----------
        user_ids : list of int, user ids to predict
        item_ids : list of int, item ids to predict
        Returns
        -------
        scores : list of float, predicted scores
        """
        vali_input = torch.tensor(self.train_mat[user_ids]).long()
        vali_input = vali_input.unsqueeze(0)
        candidate_items = torch.tensor(candidate_items).long()
        if torch.cuda.is_available():
            vali_input = vali_input.cuda()
            candidate_items =  candidate_items.cuda()
        self.eval()
        with torch.no_grad():
            pred_output = self.forward(vali_input)
            scores = pred_output[:, candidate_items]
        return scores
    
    def build_vali_train(self, vali_mat):
        user_dict = {}
        for user_id, items in vali_mat.groupby('user'):
            user_dict[int(user_id)] = []
            for idx, row in items.iterrows():
                user_dict[int(user_id)].append(int(row['item']))
                # self.train_mat[user][row['item']] = 0
        return user_dict