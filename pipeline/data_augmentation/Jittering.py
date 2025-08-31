import torch
import sys 
import os 
import pandas as pd
import importlib
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from pipeline.DL_class import FeatureVectorBuilder,DatesVerifFeatureVect


class JitteringObject(object):
    """
    A class used for injecting noise into data for augmentation.
    """

    def __init__(self, normalizers, step_ahead, H, D, W, Day_nb_steps, Week_nb_steps, shift_from_first_elmt, time_step_per_hour, init_noise_type = 'gaussian',dataset_name=None):
        """
        Initialize the JitteringObject with required parameters.
        """
        self.normalizers = normalizers
        self.step_ahead = step_ahead
        self.H = H
        self.D = D
        self.W = W
        self.Day_nb_steps = Day_nb_steps
        self.Week_nb_steps = Week_nb_steps
        self.shift_from_first_elmt = shift_from_first_elmt
        self.time_step_per_hour = time_step_per_hour
        self.mask_seq_3d = None
        self.init_noise_type = init_noise_type

        module_path = f"..load_inputs.{dataset_name}"
        module = importlib.import_module(module_path)
        self.USELESS_DATES = module.USELESS_DATES

    def compute_noise_injection(self, U_train_copy, Utarget_train_copy, ds, dataset_name, mask_inject, out_dim, alpha):
            '''
            This method retrieves a noise DataFrame for each spatial unit, reindexes it to include missing dates,
            and constructs a noise feature vector consistent with the way U_train/Utarget_train are built.
            It then applies a user-defined noise distribution (scaled and normalized) and injects the noise into
            U_train/Utarget_train to produce data-augmented tensors.

            Returns:
            --------
            U_train_copy (torch.Tensor): Augmented version of the training feature tensor
            Utarget_train_copy (torch.Tensor): Augmented version of the target feature tensor
            '''
            n, N, L = U_train_copy.shape
            self.mask_seq_3d = mask_inject.unsqueeze(-1).expand(-1, -1, L + out_dim)
            self.start = ds.tensor_limits_keeper.df_verif_train.min().min()
            self.end = ds.tensor_limits_keeper.df_verif_train.max().max()

            # 1) Retrieve and reindex noise DataFrame
            df_noises = ds.noises[dataset_name]
            noise_tensor = self._prepare_noise_tensor_from_df(df_noises)

            # 2) Build noise feature/target vectors
            U_noise, Utarget_noise = self._build_noise_feature_vectors(noise_tensor)
            
            # 3) Mask invalid dates
            amp_values, amp_values_target = self._apply_mask(U_noise, Utarget_noise, ds)

            # 4) Generate scaled noise
            scaled_noise = self._generate_scaled_noise(n, N, L, amp_values, amp_values_target, alpha,dataset_name)

            # 5) Inject noise into U_train/Utarget_train
            U_train_copy, Utarget_train_copy = self._inject_noise(
                U_train_copy, Utarget_train_copy, scaled_noise, out_dim
            )

            return U_train_copy, Utarget_train_copy


    def _prepare_noise_tensor_from_df(self, df_noises):
        '''
        Reindex df_noises to the full time range, filling missing dates
        and setting noise to 0 for closed-hour periods.
        '''
        time_freq = f'{60 // self.time_step_per_hour}min'
        self.df_noises_dates = pd.DataFrame(pd.date_range(self.start, self.end, freq=time_freq),columns=['date'])
        df_noises_reindexed = df_noises.reindex(list(self.df_noises_dates['date']))

        df_noises_reindexed[df_noises_reindexed.index.hour.isin(self.USELESS_DATES['hour'])] = 0
        if len(self.USELESS_DATES['weekday']) > 0 and len(self.USELESS_DATES['hour']) < 1:
            raise NotImplementedError(
                'NaN values occur when predicting e.g. Monday at 00:15 with Sunday 23:45/23:30/23:15, etc.'
            )
        tensor_noises_reindexed = torch.from_numpy(df_noises_reindexed.values).float()
        return tensor_noises_reindexed


    def _build_noise_feature_vectors(self, noise_tensor):
        '''
        Build noise feature and target vectors the same way as U_train/Utarget_train.
        '''
        featurevectorbuilder = FeatureVectorBuilder(
            self.step_ahead, self.H, self.D, self.W,
            self.Day_nb_steps, self.Week_nb_steps,
            self.shift_from_first_elmt
        )
        featurevectorbuilder.build_feature_vect(noise_tensor)
        featurevectorbuilder.build_target_vect(noise_tensor)
        return featurevectorbuilder.U, featurevectorbuilder.Utarget


    def _apply_mask(self, U_noise, Utarget_noise, ds):
        '''
        Filter out invalid indices so noise matches the valid training/target data.
        '''
        dates_verif_object = DatesVerifFeatureVect(self.df_noises_dates, Weeks = self.W, Days = self.D, historical_len = self.H, step_ahead = self.step_ahead, time_step_per_hour = self.time_step_per_hour)
        dates_verif_object.get_df_verif(ds.invalid_dates)
        forbidden_indice_U_noise = dates_verif_object.forbidden_indice_U

        mask_indices = [idx for idx in range(U_noise.shape[0]) if idx not in forbidden_indice_U_noise]

        return U_noise[mask_indices], Utarget_noise[mask_indices]


    def _generate_scaled_noise(self, n, N, L, amp_values, amp_values_target, alpha,dataset_name):
        '''
        Generate and normalize noise scaled by amplitude values and a user-defined alpha factor.
        '''
        # Select type of Initial Noise: 
        if self.init_noise_type == 'gaussian':
            init_noise = torch.randn(n, N, L + amp_values_target.size(-1))
        elif self.init_noise_type == 'uniform':
            init_noise = torch.FloatTensor(n, N, L + amp_values_target.size(-1)).uniform_(-1, 1)
        else:
            raise NotImplementedError

        raw_noise = init_noise * alpha * torch.cat([amp_values, amp_values_target], -1)
        return self.normalizers[dataset_name].normalize_tensor(raw_noise, feature_vect=True)


    def _inject_noise(self, U_train_copy, Utarget_train_copy, scaled_noise, out_dim):
        '''
        Inject scaled noise into the training and (optional) target tensors.
        '''
        U_train_copy = U_train_copy + (scaled_noise[..., :-out_dim] * self.mask_seq_3d[..., :-out_dim])
        if Utarget_train_copy is not None:
            Utarget_train_copy = Utarget_train_copy + (scaled_noise[..., -out_dim:] * self.mask_seq_3d[..., -out_dim:])
        return U_train_copy, Utarget_train_copy