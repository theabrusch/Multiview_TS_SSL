import numpy as np

class cpc_data_simulator():
    def __init__(self, n_sources, groups_of_dep_var, n_states, sigma, fs, length):
        self.n_sources = n_sources if isinstance(n_sources, list) else groups_of_dep_var*[n_sources]
        self.groups_of_dep_var = groups_of_dep_var
        self.sigma = sigma
        self.fs = fs
        self.length = length
        self.n_settings = n_states
        self.source_frequencies = np.zeros((n_states, np.sum(n_sources)))
        self.emission_matrix = np.zeros((np.sum(groups_of_dep_var), np.sum(n_sources)))
        k = 0
        j = 0
        for source, group in zip(n_sources, groups_of_dep_var):
            self.emission_matrix[j:j+group, k:k+source] = np.random.normal(0, 1, (group, source))
            self.source_frequencies[:, k:k+source] = np.random.uniform(1, 50, (n_states, source))
            j+=group
            k+=source
    
    def generate(self, n_samples, return_sources = False, shuffle_variables = True):
        # Generate the independent variables
        t = np.repeat(np.expand_dims(np.arange(0, self.length, 1/self.fs), axis = 0), n_samples, axis = 0)
        # Generate a random phase shift per sample
        phase_shift = np.expand_dims(np.random.uniform(0, 2*np.pi, n_samples), 1)
        # Generate the dependent variables
        x = np.zeros((n_samples, round(self.length*self.fs) , np.sum(self.groups_of_dep_var)))
        states = np.random.randint(0, self.n_settings, n_samples)

        if return_sources:
            sources = np.zeros((n_samples, round(self.length*self.fs), np.sum(self.n_sources)))

        for k in range(np.sum(self.n_sources)):
            x += np.expand_dims(np.sin(np.expand_dims(self.source_frequencies[states, k],1) * t + phase_shift), 2) @ np.expand_dims(self.emission_matrix[:, k], 0)
            if return_sources:
                sources[:, :, k] = np.sin(np.expand_dims(self.source_frequencies[states, k],1) * t + phase_shift)

        # Add noise
        x += np.random.normal(0, self.sigma, (n_samples, round(self.length*self.fs), np.sum(self.groups_of_dep_var)))
        # randomly shuffle the variables
        idx = np.arange(np.sum(self.groups_of_dep_var))
        np.random.shuffle(idx)
        if return_sources:
            if shuffle_variables:
                return x[:, :, idx], sources, self.emission_matrix[:, idx]
            return x, sources, self.emission_matrix
        else:
            return x[:, :, idx]
        

class multiview_data_simulator():
    def __init__(self, n_sources, n_variables, n_states, sigma, fs, length):
        self.n_sources = n_sources 
        self.n_variables = n_variables
        self.sigma = sigma
        self.fs = fs
        self.length = length
        self.n_settings = n_states
        self.source_frequencies = np.random.uniform(1, 50, (n_states, n_sources))
        self.emission_matrix = np.random.normal(0, 1, (n_sources, n_variables))
    
    def generate(self, n_samples, return_sources = False):
        # Generate the independent variables
        t = np.expand_dims(np.arange(0, self.length/2, 1/self.fs),0)
        t_len = t.shape[1]
        # Generate a random phase shift per sample
        phase_shift = np.expand_dims(np.random.uniform(0, 2*np.pi, n_samples), 1)
        # Generate the dependent variables
        x = np.zeros((n_samples, t_len*2, self.n_variables))

        if return_sources:
            sources = np.zeros((n_samples, round(self.length*self.fs), np.sum(self.n_sources)))

        for n in range(n_samples):
            states = np.random.choice(np.arange(0, self.n_settings), 2, replace = False)
            x[n,:t_len,:] = np.sin(np.expand_dims(self.source_frequencies[states[0]],1) * t + phase_shift[n]).T @ self.emission_matrix
            x[n,t_len:,:] = np.sin(np.expand_dims(self.source_frequencies[states[1]],1) * t + phase_shift[n]).T @ self.emission_matrix
            if return_sources:
                sources[n,:t_len,:]= np.sin(np.expand_dims(self.source_frequencies[states[0]],1) * t + phase_shift[n]).T
                sources[n,t_len:,:]= np.sin(np.expand_dims(self.source_frequencies[states[1]],1) * t + phase_shift[n]).T

        # Add noise
        x += np.random.normal(0, self.sigma, (n_samples, round(self.length*self.fs), self.n_variables))

        if return_sources:
            return x, sources, self.emission_matrix
        else:
            return x
        
class finetuning_simulator():
    def __init__(self, finetune_setup, n_sources, groups_of_dep_var, n_states, sigma, fs, length):
        self.n_sources = n_sources if isinstance(n_sources, list) else groups_of_dep_var*[n_sources]
        self.finetune_setup = finetune_setup
        self.groups_of_dep_var = groups_of_dep_var
        self.sigma = sigma
        self.fs = fs
        self.length = length
        self.n_settings = n_states
        self.var_idx = np.arange(np.sum(groups_of_dep_var))
        np.random.shuffle(self.var_idx)
        if self.finetune_setup == 'simulated_cpc':
            self.y_state = np.random.randint(n_sources[0], np.sum(self.n_sources))

        np.random.seed(42)
        em_matrix = np.random.normal(0, 1, (np.sum(groups_of_dep_var), np.sum(n_sources)))
        np.random.seed(42)
        self.source_frequencies = np.random.uniform(1, 50, (n_states, np.sum(n_sources)))

        k = 0
        j = 0
        self.emission_matrix = np.zeros((np.sum(groups_of_dep_var), np.sum(n_sources)))
        for source, group in zip(n_sources, groups_of_dep_var):
            self.emission_matrix[j:j+group, k:k+source] = em_matrix[j:j+group, k:k+source]
            j+=group
            k+=source
    
    def generate(self, n_samples):
        # Generate the independent variables
        t = np.repeat(np.expand_dims(np.arange(0, self.length, 1/self.fs), axis = 0), n_samples, axis = 0)
        # Generate a random phase shift per sample
        phase_shift = np.expand_dims(np.random.uniform(0, 2*np.pi, n_samples), 1)
        # Generate the dependent variables
        x = np.zeros((n_samples, round(self.length*self.fs) , np.sum(self.groups_of_dep_var)))
        if self.finetune_setup == 'simulated_multiview':
            states = np.random.randint(0, self.n_settings, n_samples)
        else:
            states = np.random.randint(0, self.n_settings, (n_samples, np.sum(self.n_sources)))

        for k in range(np.sum(self.n_sources)):
            if self.finetune_setup == 'simulated_multiview':
                s = states
            else:
                s = states[:,k]
            x += np.expand_dims(np.sin(np.expand_dims(self.source_frequencies[s, k],1) * t + phase_shift), 2) @ np.expand_dims(self.emission_matrix[:, k], 0)
        
        # Add noise
        x += np.random.normal(0, self.sigma, (n_samples, round(self.length*self.fs), np.sum(self.groups_of_dep_var)))

        if self.finetune_setup == 'simulated_cpc':
            states = states[:, self.y_state]

        return x[:, :, self.var_idx], states