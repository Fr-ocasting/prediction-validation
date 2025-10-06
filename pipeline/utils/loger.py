class LOG(object):
    def __init__(self):
        self.log_final = ''

    def add_log(self, test_metrics, metrics, trial_id, step_ahead,horizon_step):
        # Construction de la partie "All Steps"
        all_parts = [
            f"{m.upper()} = {test_metrics[f'{m}_all']:.3f}"
            for m in metrics
        ]
        log_final_i = "All Steps " + ", ".join(all_parts)
        
        # On stocke ou on concatène dans self.log_final
        if not hasattr(self, 'log_final'):
            self.log_final = ""
        self.log_final += f"{trial_id}:   {log_final_i}\n"
        
        # Affichage du résumé global
        print(f"\n--------- Test ---------\n{log_final_i}")
        
        # Affichage pas-à-pas pour h allant de 1 à step_ahead
        for h in range(horizon_step, step_ahead + 1,horizon_step):
            step_parts = [
                f"{m.upper()} = {test_metrics[f'{m}_h{h}']:.3f}"
                for m in metrics
            ]
            print(f"Step {h} " + ", ".join(step_parts))

    def display_log(self):
        print(self.log_final)