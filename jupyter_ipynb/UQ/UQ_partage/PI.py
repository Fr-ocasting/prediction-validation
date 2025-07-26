import torch
import numpy as np

class Calibrator:
    def __init__(self, alpha, device):
        self.alpha = alpha
        self.device = device
        self.Q = None # Le terme de calibration sera stocké ici

    def get_predictions(self, trainer):
        """Récupère les prédictions du modèle sur le jeu de calibration."""
        # La méthode predict du trainer retourne déjà les prédictions et les vraies valeurs
        preds, y_calib = trainer.predict(mode='calib')
        # S'assurer que les prédictions sont triées (q_low <= q_high)
        preds, _ = torch.sort(preds, dim=-1)
        
        self.y_calib = y_calib.squeeze().to(self.device)
        if preds.size(-1) == 1: 
            self.lower_q = preds[..., 0].to(self.device)
            self.upper_q = preds[..., 0].to(self.device)
        else:
            self.lower_q = preds[..., 0].to(self.device)
            self.upper_q = preds[..., 1].to(self.device)

    def get_conformity_scores(self):
        """Calcule les scores de non-conformité."""
        # E_i = max(q_low(X_i) - Y_i, Y_i - q_high(X_i))
        scores = torch.max(self.lower_q - self.y_calib, self.y_calib - self.upper_q)
        self.conformity_scores = scores

    def get_quantile(self):
        """Calcule le quantile des scores pour la calibration."""
        n = self.y_calib.size(0)
        # Niveau de quantile ajusté pour la calibration
        quantile_order = np.ceil((1 - self.alpha) * (n + 1)) / n
        self.quantile_order = min(quantile_order, 1.0) # S'assurer que le quantile ne dépasse pas 1

        # Calculer le quantile sur toutes les observations et stations
        
        # Le quantile est calculé en utilisant la méthode 'higher' pour être conservateur
        self.Q = torch.quantile(self.conformity_scores, self.quantile_order, interpolation='higher',dim = 0)


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
            Q = Q_tensor

        self.Q = Q.detach().cpu() if Q is not None else None


        
        if type_calib == 'CQR':
            self.bands = {'lower':preds[...,0].unsqueeze(-1)-self.Q, 'upper': preds[...,1].unsqueeze(-1)+self.Q}
            self.lower = preds[...,0].unsqueeze(-1)-self.Q
            self.upper = preds[...,1].unsqueeze(-1)+self.Q

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