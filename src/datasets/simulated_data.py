import numpy as np

class multivariate_data_simulator():
    def __init__(self, n_sources, groups_of_dep_var, n_states, sigma, fs, length):
        self.n_sources = n_sources if isinstance(n_sources, list) else groups_of_dep_var*[n_sources]
        self.groups_of_dep_var = groups_of_dep_var
        self.sigma = sigma
        self.fs = fs
        self.length = length
        self.n_settings = n_states
        self.source_frequencies = []
        self.emission_matrix = []
        for source, group in zip(n_sources, groups_of_dep_var):
            setting_freq = np.zeros((n_states, source))
            self.emission_matrix.append(np.random.normal(0, 1, (group, source)))
            for i in range(n_states):
                setting_freq[i,:] = np.random.uniform(1, 50, source)
            self.source_frequencies.append(setting_freq)
    
    def generate(self, n_samples, return_sources = False):
        # Generate the independent variables
        t = np.repeat(np.expand_dims(np.arange(0, self.length, 1/self.fs), axis = 0), n_samples, axis = 0)
        # Generate a random phase shift per sample
        phase_shift = np.expand_dims(np.random.uniform(0, 2*np.pi, n_samples), 1)
        # Generate the dependent variables
        y = np.zeros((n_samples, round(self.length*self.fs) , np.sum(self.groups_of_dep_var)))
        states = np.random.randint(0, self.n_settings, n_samples)

        if return_sources:
            assert len(self.groups_of_dep_var) == 1, "return_sources only works for a single group of sources"
            sources = np.zeros((n_samples, round(self.length*self.fs), self.n_sources[0]))
            emission_matrices = np.zeros((n_samples, self.groups_of_dep_var[0], self.n_sources[0]))

        j = 0
        i = 0
        for group in self.groups_of_dep_var:
            for k in range(self.n_sources[i]):
                y[:, :, j:j+group] += np.expand_dims(np.sin(np.expand_dims(self.source_frequencies[i][states, k],1) * t + phase_shift), 2) @ np.expand_dims(self.emission_matrix[i][:, k], 0)
                if return_sources:
                    sources[:, :, k] = np.sin(np.expand_dims(self.source_frequencies[i][states, k],1) * t + phase_shift)
                    emission_matrices[:, :, k] = self.emission_matrix[i][:, k]
            i+=1
            j+=group
        # Add noise
        y += np.random.normal(0, self.sigma, (n_samples, round(self.length*self.fs), np.sum(self.groups_of_dep_var)))
        # randomly shuffle the variables
        if return_sources:
            return y, sources, emission_matrices
        else:
            idx = np.arange(np.sum(self.groups_of_dep_var))
            np.random.shuffle(idx)
            return y[:, :, idx]