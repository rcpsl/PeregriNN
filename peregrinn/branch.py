from peregrinn.verifier import VerificationResult


class Branch:
    '''
    Defines a computational branch used in verification
    '''
    def __init__(self, input_bounds, spec) -> None:
        self.input_bounds = input_bounds
        self.spec = spec
        self.fixed_neurons = [] #list of tuples (layer_idx, neuron_idx, phase)
        self.verification_result = VerificationResult.UNKNOWN

    def fix_neuron(self, layer_idx, neuron_idx, phase):
        self.fixed_neurons.append((layer_idx, neuron_idx, phase))

    def clone(self):
        new_branch = Branch(self.input_bounds, self.spec)
        new_branch.fixed_neurons = self.fixed_neurons.copy()
        return new_branch    

