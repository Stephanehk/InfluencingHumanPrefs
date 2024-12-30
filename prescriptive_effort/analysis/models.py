import torch

class LogisticRegression(torch.nn.Module):
    '''
    A logistic regression model for predicting preferences
    '''
    def __init__(self, input_size, bias, prob_uniform_resp=True):
        '''
        Input:
        - input_size: the number of inputs to consider
        - bias: the initial network bias
        - prob_uniform_resp: if true, scale the predicted preference probabilities by the probaility of a uniform response
        '''
        super(LogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 1,bias=bias)
        self.c = torch.nn.Parameter(torch.tensor([0.0]))
        self.prob_uniform_resp = prob_uniform_resp
        #set initial weights to 0 for readability
        with torch.no_grad():
            self.linear1.weight = torch.nn.Parameter(torch.tensor([[0 for i in range(input_size)]],dtype=torch.float))
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear1(x))
        if not self.prob_uniform_resp:
            return y_pred
        sig_c = torch.sigmoid(self.c)
        scaled_pred = torch.add(torch.mul(torch.sub(1,sig_c), y_pred),torch.divide(sig_c,2))
        return scaled_pred