import torch 

class PI_object(object):
    def __init__(self,preds,Y_true,alpha = None, type_calib = 'CQR',Q = None,T_labels = None):
        super(PI_object,self).__init__()
        self.alpha = alpha
        self.Y_true = Y_true

        if type(Q) == dict:
            Q_tensor = torch.zeros(preds.size(0),preds.size(1),1).to(preds)
            for label in T_labels.unique():
                indices = torch.nonzero(T_labels == label).squeeze()
                try: 
                    Q_tensor[indices,:,0] = Q[label.item()]['Q'][0,:,0]
                except:
                    print(f"No Conformal Calibration value found for {label.item()}. Will be set to 100") 
                    Q_tensor[indices,:,0] = 100
        else : 
            Q_tensor = Q

        self.Q_tensor = Q_tensor
        
        if type_calib == 'CQR':
            self.bands = {'lower':preds[...,0].unsqueeze(-1)-self.Q_tensor, 'upper': preds[...,1].unsqueeze(-1)+self.Q_tensor}
            self.lower = preds[...,0].unsqueeze(-1)-self.Q_tensor
            self.upper = preds[...,1].unsqueeze(-1)+self.Q_tensor

        if type_calib =='classic':
            self.bands = {'lower':preds[...,0].unsqueeze(-1), 'upper': preds[...,1].unsqueeze(-1)}
            self.lower = preds[...,0].unsqueeze(-1)
            self.upper = preds[...,1].unsqueeze(-1)

        self.MPIW()
        self.PICP()
        
    def MPIW(self):
        self.mpiw = torch.mean(self.bands['upper']-self.bands['lower']).item()
        return(self.mpiw)
        
    def PICP(self):
        self.picp = torch.sum((self.lower<self.Y_true)&(self.Y_true<self.upper)).item()/torch.prod(torch.Tensor([s for s in self.lower.size()])).item()
        return(self.picp)