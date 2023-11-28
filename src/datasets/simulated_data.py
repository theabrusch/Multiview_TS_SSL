import numpy as np

class pretraining_data_simulator():
    def __init__(self, n_sources, groups_of_dep_var, sigma, fs, length, simulator_type, normalize_emission = False, seed = 42):
        self.n_sources = n_sources if isinstance(n_sources, list) else groups_of_dep_var*[n_sources]
        self.groups_of_dep_var = groups_of_dep_var
        self.sigma = sigma
        self.fs = fs
        self.length = length
        self.simulator_type = simulator_type
        self.seed = seed
        self.normalize_emission = normalize_emission
        self.var_idx = np.arange(np.sum(self.groups_of_dep_var))
        np.random.shuffle(self.var_idx)

        np.random.seed(self.seed)
        self.emission_matrix = self.generate_emission_matrix(self.seed)

    def generate_emission_matrix(self, seed = None):
        np.random.seed(seed)
        em_matrix = np.random.normal(0, 1, (np.sum(self.n_sources), np.sum(self.groups_of_dep_var)))
        k = 0
        j = 0
        emission_matrix = np.zeros((np.sum(self.n_sources), np.sum(self.groups_of_dep_var)))
        for source, group in zip(self.n_sources, self.groups_of_dep_var):
            emission_matrix[k:k+source, j:j+group] = em_matrix[k:k+source, j:j+group]
            j+=group
            k+=source
        if self.normalize_emission:
            exp_emission_matrix = np.exp(emission_matrix)
            emission_matrix = exp_emission_matrix / np.sum(exp_emission_matrix, axis = 0, keepdims = True)
        np.random.seed(None)
        return emission_matrix
    
    def generate(self, n_samples, random_emission_matrix = False, return_sources = False, shuffle_variables = True):
        # Generate the independent variables
        if self.simulator_type == 'simulated_cpc':
            t = np.expand_dims(np.arange(0, self.length, 1/self.fs),0)
        else:
            t = np.expand_dims(np.arange(0, self.length/2, 1/self.fs),0)
        t_len = t.shape[1]
        # Generate a random phase shift per sample
        phase_shift = np.expand_dims(np.random.uniform(0, 2*np.pi, n_samples), 1)
        # Generate the dependent variables
        x = np.zeros((n_samples, round(self.length*self.fs) , np.sum(self.groups_of_dep_var)))
        np.random.seed(None)
        if return_sources:
            sources = np.zeros((n_samples, round(self.length*self.fs), np.sum(self.n_sources)))
            freqs = np.zeros((n_samples, np.sum(self.n_sources)))

        for n in range(n_samples):
            if random_emission_matrix:
                em_matrix = self.generate_emission_matrix()
            else:
                em_matrix = self.emission_matrix
            
            if self.simulator_type == 'simulated_cpc':
                freqs_1 = np.random.uniform(1, (self.fs/2), (np.sum(self.n_sources), 1))
                freqs_2 = freqs_1
                source = np.sin(freqs_1 * t + phase_shift[n]).T
                
            else:
                freqs_1 = np.random.uniform(1, (self.fs/2)/2, (np.sum(self.n_sources), 1))
                freqs_2 = np.random.uniform((self.fs/2)/2, self.fs/2, (np.sum(self.n_sources), 1))
                source = np.zeros((t_len*2, np.sum(self.n_sources)))
                source[:t_len,:] = np.sin(freqs_1 * t + phase_shift[n]).T
                source[t_len:,:] = np.sin(freqs_2 * t + phase_shift[n]).T
            x[n,:,:] = source @ em_matrix
            
            if return_sources:
                sources[n,:,:]= source
                freqs[n,:] = np.concatenate(freqs_1, axis = 0).flatten()

        # Add noise
        x += np.random.normal(0, self.sigma, (n_samples, round(self.length*self.fs), np.sum(self.groups_of_dep_var)))
        # randomly shuffle the variables
        if return_sources:
            if shuffle_variables:
                return x[:, :, self.var_idx], sources, self.emission_matrix[:, self.var_idx], freqs
            return x, sources, self.emission_matrix, freqs
        else:
            return x[:, :, self.var_idx]
        

class multiview_data_simulator():
    def __init__(self, n_sources, n_variables, n_states, sigma, fs, length):
        self.n_sources = n_sources 
        self.n_variables = n_variables
        self.sigma = sigma
        self.fs = fs
        self.length = length
        self.n_settings = n_states
        self.source_frequencies = np.random.uniform(1, 50, (n_states, n_sources))
        np.random.seed(42)
        self.emission_matrix = np.random.normal(0, 1, (n_variables, n_sources))
    
    def generate(self, n_samples, random_emission_matrix = False, random_settings = False, random_freqs = False, return_sources = False):
        # Generate the independent variables
        t = np.expand_dims(np.arange(0, self.length/2, 1/self.fs),0)
        t_len = t.shape[1]
        # Generate a random phase shift per sample
        phase_shift = np.expand_dims(np.random.uniform(0, 2*np.pi, n_samples), 1)
        # Generate the dependent variables
        x = np.zeros((n_samples, t_len*2, self.n_variables))

        if return_sources:
            sources = np.zeros((n_samples, t_len*2, np.sum(self.n_sources)))
            emission_matrix = np.zeros((n_samples,self.n_variables, self.n_sources))

        for n in range(n_samples):
            if random_settings:
                states  = np.random.randint(0, self.n_settings, (2, self.n_sources))
                freqs_1 = np.random.uniform(1, (self.fs/2)/2, (self.n_sources, 1))
                freqs_2 = np.random.uniform((self.fs/2)/2, self.fs/2, (self.n_sources, 1))
            else:
                states = np.random.choice(np.arange(0, self.n_settings), 2, replace = False)
                freqs_1 = np.take_along_axis(self.source_frequencies, np.expand_dims(states[0], 0), axis = 0).transpose()
                freqs_2 = np.take_along_axis(self.source_frequencies, np.expand_dims(states[1], 0), axis = 0).transpose()

            if random_emission_matrix:
                em_matrix = np.random.normal(0, 1, (self.n_sources, self.n_variables)).T
            else:
                em_matrix = self.emission_matrix.T
                
            x[n,:t_len,:] = np.sin(freqs_1 * t + phase_shift[n]).T @ em_matrix
            x[n,t_len:,:] = np.sin(freqs_2 * t + phase_shift[n]).T @ em_matrix
            if return_sources:
                sources[n,:t_len,:]= np.sin(freqs_1 * t + phase_shift[n]).T
                sources[n,t_len:,:]= np.sin(freqs_2 * t + phase_shift[n]).T
                emission_matrix[n,:,:] = em_matrix

        # Add noise
        x += np.random.normal(0, self.sigma, (n_samples, round(self.length*self.fs), self.n_variables))

        if return_sources:
            return x, sources, emission_matrix
        else:
            return x
        
class finetuning_simulator():
    def __init__(self, n_sources, groups_of_dep_var, n_states, sigma, fs, length, n_state_sources = 1, normalize_emission = False, seed = 42):
        self.n_sources = n_sources if isinstance(n_sources, list) else groups_of_dep_var*[n_sources]
        self.groups_of_dep_var = groups_of_dep_var
        self.sigma = sigma
        self.fs = fs
        self.length = length
        self.n_settings = n_states
        self.normalize_emission = normalize_emission
        self.var_idx = np.arange(np.sum(groups_of_dep_var))
        self.seed = seed

        # shuffle output variables
        np.random.seed(seed)
        np.random.shuffle(self.var_idx)

        # sample the source whose state will be the dependent variable
        np.random.seed(seed)
        self.y_state = np.random.randint(0, np.sum(self.n_sources), n_state_sources)
        self.y_freqs = np.array([[5, 31], [42, 23], [12, 17], [37, 19], [29, 7], [11, 41], [2, 47], [43, 3], [13, 19], [17, 23]])
        np.random.seed(seed)
        self.source_frequencies = np.random.uniform(1, 50, (n_states, np.sum(self.n_sources)))

        self.emission_matrix = self.generate_emission_matrix(seed = self.seed)

    def generate_emission_matrix(self, seed = None):
        np.random.seed(seed)
        em_matrix = np.random.normal(0, 1, (np.sum(self.n_sources), np.sum(self.groups_of_dep_var)))
        k = 0
        j = 0
        emission_matrix = np.zeros((np.sum(self.n_sources), np.sum(self.groups_of_dep_var)))
        for source, group in zip(self.n_sources, self.groups_of_dep_var):
            emission_matrix[k:k+source, j:j+group] = em_matrix[k:k+source, j:j+group]
            j+=group
            k+=source
        if self.normalize_emission:
            exp_emission_matrix = np.exp(emission_matrix)
            emission_matrix = exp_emission_matrix / np.sum(exp_emission_matrix, axis = 0, keepdims = True)
        np.random.seed(None)
        return emission_matrix
    
    def generate(self, n_samples, return_sources = False, random_freqs = False):
        # Generate the independent variables
        t = np.expand_dims(np.arange(0, self.length, 1/self.fs),0)
        # Generate a random phase shift per sample
        phase_shift = np.expand_dims(np.random.uniform(0, 2*np.pi, n_samples), 1)
        # Generate the dependent variables
        x = np.zeros((n_samples, round(self.length*self.fs) , np.sum(self.groups_of_dep_var)))

        if return_sources:
            sources = np.zeros((n_samples, round(self.length*self.fs), np.sum(self.n_sources)))

        if random_freqs:
            states = np.random.randint(0, self.n_settings, (n_samples))
        else:
            states = np.random.randint(0, self.n_settings, (n_samples, np.sum(self.n_sources)))

        #for k in range(np.sum(self.n_sources)):
        #    s = states[:,k]
        #    x += np.expand_dims(np.sin(np.expand_dims(self.source_frequencies[s, k],1) * t + phase_shift), 2) @ np.expand_dims(self.emission_matrix[:, k], 0)
        #    if return_sources:
        #        sources[:,:,k] = np.sin(np.expand_dims(self.source_frequencies[s, k],1) * t + phase_shift)

        for n in range(n_samples):
            if random_freqs:
                freqs = np.random.uniform(1, (self.fs/2), (np.sum(self.n_sources), 1))
                for state in self.y_state:
                    freqs[state] = self.y_freqs[state, states[n]]
            else:
                freqs = np.take_along_axis(self.source_frequencies, np.expand_dims(states[n], 0), axis = 0).transpose()
            source = np.sin(freqs * t + phase_shift[n]).T
            x[n,:,:] = source @ self.emission_matrix
            
            if return_sources:
                sources[n,:,:]= source

        
        # Add noise
        x += np.random.normal(0, self.sigma, (n_samples, round(self.length*self.fs), np.sum(self.groups_of_dep_var)))

        if not random_freqs:
            states = states[:, self.y_state]

        if return_sources:
            return x[:, :, self.var_idx], sources, self.emission_matrix[:, self.var_idx], states
        else:
            return x[:, :, self.var_idx], states