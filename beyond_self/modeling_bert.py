from __future__ import absolute_import, division, print_function
from pytorch_transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch.nn.functional as F
import torch
import trick.focal_loss as focal_loss


class Bert_Match(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        outputs = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                            token_type_ids=batch_seq_segments)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
    
class BERT_Match_FocalLoss(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        outputs = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                            token_type_ids=batch_seq_segments)
        for parameter in self.bert.parameters():
            parameter.requires_grad = True
        pooled_output = outputs[1]
        #pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = focal_loss.FocalLoss(gamma=2, alpha=0.5)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities




class BERT_Match_GRU(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        for parameter in self.bert.parameters():
            parameter.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        '''
        self.gru = []
        for i in range(args.gru_layers):
            self.gru.append(
                nn.GRU(config.hidden_size if i == 0 else args.gru_hidden_size * 4, args.gru_hidden_size, num_layers=1,
                       bidirectional=True, batch_first=True).cuda())
        self.gru = nn.ModuleList(self.gru)
        '''
        self.gru = nn.GRU(config.hidden_size, args.gru_hidden_size, num_layers=2, bidirectional=True, batch_first=True).cuda()

        #self.classifier = nn.Linear(args.gru_hidden_size * 2, self.config.num_labels)
        self.classifier = nn.Sequential(
            nn.Linear(args.gru_hidden_size * 4, args.gru_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(args.gru_hidden_size * 2, self.config.num_labels)
        )

        self.init_weights()

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        outputs = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                            token_type_ids=batch_seq_segments)
        last_hidden_state = outputs[0]
        '''
        for gru in self.gru:
            try:
                gru.flatten_parameters()
            except:
                pass
            output, h_n = gru(last_hidden_state)
        '''
        output, h_n = self.gru(last_hidden_state)
        x = h_n.permute(1, 0, 2).reshape(batch_seqs.size(0), -1).contiguous()
        x = self.dropout(x)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

class BERT_Match_LSTM(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.GRU(config.hidden_size, args.gru_hidden_size, num_layers=2, bidirectional=True, batch_first=True).cuda()
        self.fc_rnn = nn.Linear(args.gru_hidden_size * 2, config.num_labels)

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        outputs = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                            token_type_ids=batch_seq_segments)
        last_hidden_state = outputs[0]
        output, h_n = self.lstm(last_hidden_state)
        x = output[:, -1, :]
        x = self.dropout(x)
        logits = self.fc_rnn(x)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

class BERT_Match_RCNN(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.LSTM(config.hidden_size, args.rnn_hidden_size, args.num_layers, bidirectional=True, batch_first=True)
        self.maxpool = nn.MaxPool1d(args.pad_size)
        self.fc = nn.Linear(args.rnn_hidden_size *2 + config.hidden_size, args.num_layers)

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        outputs = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                            token_type_ids=batch_seq_segments)
        encoder_out = outputs[0]
        out, _ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1).contiguous()
        out = self.maxpool(out).squeeze()
        logits = self.fc(out)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class BERT_MATCH_DPCNN(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        for parameter in self.bert.parameters():
            parameter.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.conv_region = nn.Conv2d(1, args.num_filters, (3, config.hidden_size), stride=1)
        self.conv = nn.Conv2d(args.num_filters, args.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(args.num_filters, config.num_labels)

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        outputs = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                            token_type_ids=batch_seq_segments)
        x = outputs[0]
        x = x.unsqueeze(1) # # [batch_size, hidden_size, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, hidden_size, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, hidden_size, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, hidden_size, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, hidden_size, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, hidden_size, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.dropout(x)
        logits = self.fc(x)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

class BERT_MATCH_Siamse(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        for parameter in self.bert.parameters():
            parameter.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)