"""
Mathematical Expression Vocabulary Management
Handles tokenization and encoding of mathematical symbols and expressions
"""

class MathVocabulary:
    """Vocabulary for mathematical expressions"""
    
    def __init__(self):
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        
        # Build base vocabulary with mathematical symbols
        self.base_tokens = [
            # Special tokens (MUST be first 4)
            self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN,
            
            # Numbers
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            
            # Basic operators
            '+', '-', '=', '(', ')', '[', ']', '{', '}',
            '\\times', '\\cdot', '\\div', '/', '*',
            
            # Variables (lowercase)
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            
            # Variables (uppercase) - common in math
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            
            # Greek letters (LaTeX format - common in CROHME)
            '\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\varepsilon',
            '\\zeta', '\\eta', '\\theta', '\\vartheta', '\\iota', '\\kappa',
            '\\lambda', '\\mu', '\\nu', '\\xi', '\\pi', '\\varpi', '\\rho',
            '\\varrho', '\\sigma', '\\varsigma', '\\tau', '\\upsilon', '\\phi',
            '\\varphi', '\\chi', '\\psi', '\\omega',
            
            # Greek uppercase
            '\\Gamma', '\\Delta', '\\Theta', '\\Lambda', '\\Xi', '\\Pi',
            '\\Sigma', '\\Upsilon', '\\Phi', '\\Psi', '\\Omega',
            
            # Mathematical functions
            '\\sin', '\\cos', '\\tan', '\\sec', '\\csc', '\\cot',
            '\\arcsin', '\\arccos', '\\arctan',
            '\\sinh', '\\cosh', '\\tanh',
            '\\log', '\\ln', '\\lg', '\\exp',
            
            # Calculus and advanced math
            '\\int', '\\sum', '\\prod', '\\lim', '\\partial', '\\nabla',
            '\\sqrt', '\\frac', '^', '_',
            
            # Relations and comparisons
            '\\leq', '\\geq', '\\neq', '\\equiv', '\\approx', '\\sim',
            '\\lt', '\\gt', '\\in', '\\notin', '\\subset', '\\supset',
            
            # Special symbols
            '\\infty', '\\pm', '\\mp', '\\circ', '\\degree',
            '\\parallel', '\\perp', '\\angle',
            
            # Dots and spacing
            '\\ldots', '\\cdots', '\\vdots', '\\ddots',
            '\\ ',  # LaTeX space
            
            # Common fractions
            '\\frac{1}{2}', '\\frac{1}{3}', '\\frac{1}{4}',
            
            # Punctuation that might appear
            '.', ',', '!', '?', ';', ':',
        ]
        
        # Create token mappings
        self.token_to_id = {token: i for i, token in enumerate(self.base_tokens)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        # Special token IDs for easy access
        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.sos_id = self.token_to_id[self.SOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]
        
    def encode(self, tokens):
        """Convert tokens to IDs"""
        if isinstance(tokens, str):
            # If string passed, assume it's already tokenized LaTeX
            tokens = self.simple_tokenize(tokens)
        
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]
    
    def decode(self, ids, remove_special=True):
        """Convert IDs to tokens"""
        tokens = [self.id_to_token.get(id, self.UNK_TOKEN) for id in ids]
        
        if remove_special:
            # Remove special tokens
            special_tokens = {self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN}
            tokens = [t for t in tokens if t not in special_tokens]
        
        return tokens
    
    def decode_to_string(self, ids, remove_special=True):
        """Convert IDs to LaTeX string"""
        tokens = self.decode(ids, remove_special)
        return ''.join(tokens)  # Simple concatenation for now
    
    def simple_tokenize(self, latex_string):
        """
        Simple LaTeX tokenizer
        This is basic - you might want a more sophisticated one later
        """
        tokens = []
        i = 0
        
        while i < len(latex_string):
            if latex_string[i] == '\\':
                # LaTeX command
                j = i + 1
                while j < len(latex_string) and latex_string[j].isalpha():
                    j += 1
                command = latex_string[i:j]
                
                # Handle special cases like \frac{...}{...}
                if command == '\\frac' and j < len(latex_string) and latex_string[j] == '{':
                    # Find the full fraction
                    brace_count = 0
                    k = j
                    numerator_start = k + 1
                    numerator_end = None
                    denominator_start = None
                    denominator_end = None
                    
                    while k < len(latex_string):
                        if latex_string[k] == '{':
                            brace_count += 1
                        elif latex_string[k] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                if numerator_end is None:
                                    numerator_end = k
                                    # Look for second part
                                    if k + 1 < len(latex_string) and latex_string[k + 1] == '{':
                                        denominator_start = k + 2
                                        brace_count = 1
                                    else:
                                        break
                                else:
                                    denominator_end = k
                                    break
                        k += 1
                    
                    if numerator_end and denominator_end:
                        tokens.append('\\frac')
                        tokens.append('{')
                        # Recursively tokenize numerator
                        num_tokens = self.simple_tokenize(latex_string[numerator_start:numerator_end])
                        tokens.extend(num_tokens)
                        tokens.append('}')
                        tokens.append('{')
                        # Recursively tokenize denominator
                        den_tokens = self.simple_tokenize(latex_string[denominator_start:denominator_end])
                        tokens.extend(den_tokens)
                        tokens.append('}')
                        i = k + 1
                    else:
                        tokens.append(command)
                        i = j
                else:
                    tokens.append(command)
                    i = j
            else:
                # Single character
                tokens.append(latex_string[i])
                i += 1
        
        return tokens
    
    def add_tokens(self, new_tokens):
        """Add new tokens to vocabulary"""
        for token in new_tokens:
            if token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[token] = new_id
                self.id_to_token[new_id] = token
                self.base_tokens.append(token)
    
    def __len__(self):
        return len(self.token_to_id)
    
    def get_vocab_size(self):
        return len(self.token_to_id)
    
    def save_vocab(self, filepath):
        """Save vocabulary to file"""
        import json
        vocab_data = {
            'token_to_id': self.token_to_id,
            'base_tokens': self.base_tokens
        }
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    def load_vocab(self, filepath):
        """Load vocabulary from file"""
        import json
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        self.base_tokens = vocab_data['base_tokens']
        self.id_to_token = {int(i): token for token, i in self.token_to_id.items()}
        
        # Update special token IDs
        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.sos_id = self.token_to_id[self.SOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]


def test_vocabulary():
    """Test the vocabulary functionality"""
    print("🧪 Testing Math Vocabulary...")
    
    vocab = MathVocabulary()
    
    # Test basic encoding/decoding
    test_expr = "x^2 + 1"
    tokens = vocab.simple_tokenize(test_expr)
    print(f"Original: {test_expr}")
    print(f"Tokens: {tokens}")
    
    ids = vocab.encode(tokens)
    print(f"Token IDs: {ids}")
    
    decoded = vocab.decode(ids)
    print(f"Decoded: {decoded}")
    
    # Test with LaTeX
    latex_expr = "\\frac{x^2}{2} + \\sin(x)"
    latex_tokens = vocab.simple_tokenize(latex_expr)
    print(f"\nLaTeX: {latex_expr}")
    print(f"Tokens: {latex_tokens}")
    
    print(f"\nVocabulary size: {len(vocab)}")
    print("✅ Vocabulary test complete!")


if __name__ == "__main__":
    test_vocabulary()