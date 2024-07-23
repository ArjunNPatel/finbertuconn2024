import numpy as np
import torch
import torch.nn as nn
import tqdm, boto3, requests, regex, sentencepiece, sacremoses
from huggingface_hub import PyTorchModelHubMixin
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'yiyanghkust/finbert-tone')
class BertForSequenceClassification(nn.Module, PyTorchModelHubMixin):
    def __init__(self, pretrained_model_name, num_labels=3):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', pretrained_model_name)
        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.finaloutput = nn.Linear(128, num_labels)
        self.softmaxlayer = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, labels = None, *args, **kwargs):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        fc1_output = nn.functional.relu(self.fc1(pooled_output))
        logits = self.finaloutput(fc1_output)
        logits = self.softmaxlayer(logits)
        loss = None
        if labels is not None:
          loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return {"logits":logits,
                "loss": loss
                }

model = BertForSequenceClassification.from_pretrained("ANP1/finbert-tone-v0")

def score(logit):
  return logit[0]*-1 + logit[1]*0 + logit[2]*1
def magnitude(logit):
  return 1.5*np.amax(logit) - 0.5

def sentimentanalysis(myinputs):
  finalresults = []
  for i in range(len(myinputs)):
    if type(myinputs) is str:
      templist = []
      templist.append(myinputs)
      myinputs = templist
  outputs = tokenizer(myinputs, return_tensors = 'pt', padding = "max_length", truncation=True, max_length=128)
  outputs = model(**outputs)["logits"].detach().numpy()
  for row in outputs:
    logit = outputs[i,:]
    finalresults.append([myinputs[i], score(logit),magnitude(logit)])
  return finalresults
