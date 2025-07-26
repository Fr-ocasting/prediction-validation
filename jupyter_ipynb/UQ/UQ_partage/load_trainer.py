import torch
import torch.nn as nn
from PI import Calibrator


class QuantileLoss(nn.Module):
    def __init__(self,quantiles):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        # y-^y 
        errors = target - preds       #Soustraction sur la dernière dimension, à priori target 1 sortie et prediction len(quantiles) sorties

        # Errors : [B,N,2]  cause target [B,N,1] and preds [B,N,2]  
        losses = torch.max(self.quantiles*errors,(self.quantiles-1)*errors) # Récupère le plus grand des deux écart, pour chacune des estimations de quantile
        
        # Prends la moyenne de toute les erreurs
        loss = torch.mean(torch.sum(losses,dim = -1))   #  Loss commune pour toutes les stations. sinon loss par stations : torch.mean(torch.sum(losses,dim = -1),dim = 0)

        return(loss)


class Trainer:
    def __init__(self, model, dataloaders, config):
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_loss = []
        self.valid_loss = []
        
        # Fonctions de perte et optimiseur
        if config["training"]["loss_function"] == "mse":
            self.loss_fn = nn.MSELoss()
        elif config["training"]["loss_function"] == "quantile":
            if 'quantile_list' in config["training"].keys():
                quantiles = torch.Tensor(config["training"]["quantile_list"]).to(self.device)
            else:
                quantiles = torch.Tensor([config["training"]["alpha"]/2,1-config["training"]["alpha"]/2]).to(self.device)
            self.loss_fn = QuantileLoss(quantiles)
        else:
            raise ValueError(f"Fonction de perte '{config['training']['loss_function']}' non supportée.")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )

    def _run_epoch(self, dataloader, is_training=True):
        """Exécute une epoch d'entraînement ou de validation."""
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            if is_training:
                self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(is_training):
                outputs = self.model(X_batch)
                if y_batch.dim() == 2:
                    y_batch = y_batch.unsqueeze(-1)  # Assure que y_batch a la même dimension que les prédictions dans le cas de quantiles
                loss = self.loss_fn(outputs, y_batch)
            
            if is_training:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
        
        return total_loss / len(dataloader.dataset)

    def train_and_valid(self):
        """Boucle principale d'entraînement et de validation."""
        for epoch in range(self.config["training"]["epochs"]):
            train_loss = self._run_epoch(self.dataloaders["train"], is_training=True)
            valid_loss = self._run_epoch(self.dataloaders["valid"], is_training=False)
            
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

    def predict(self, mode='test',allow_dropout = False):
        """Fait des prédictions sur un ensemble de données spécifié."""
        if mode not in self.dataloaders:
            raise ValueError(f"Le mode '{mode}' n'est pas valide. "
                            f"Choisissez parmi {list(self.dataloaders.keys())}.")
        
        dataloader = self.dataloaders[mode]
        if allow_dropout:
            self.model.train()
        else:
            self.model.eval()
        predictions = []
        real_values = []
        with torch.no_grad():
            for X_batch, Y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                if Y_batch.dim() == 2:
                    Y_batch = Y_batch.unsqueeze(-1) # Assure que Y_batch a la même dimension que les prédictions dans le cas de quantiles

                predictions.append(outputs.cpu())
                real_values.append(Y_batch.cpu())
        
        return torch.cat(predictions, dim=0),torch.cat(real_values, dim=0)
    
    def conformalize_calibration(self):
        """Orchestre la calibration des intervalles de prédiction."""
        calibrator = Calibrator(self.config["training"]["alpha"], self.device)
        
        # 1. Obtenir les prédictions sur le jeu de calibration
        calibrator.get_predictions(self)
        
        # 2. Calculer les scores de non-conformité
        calibrator.get_conformity_scores()

        # 3. Calculer le quantile Q
        calibrator.get_quantile()
        
        return calibrator
    