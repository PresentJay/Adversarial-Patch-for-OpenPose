"""
Reference:
    Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""

import torch

if __name__ == '__main__':
    
    # 1. set available process units alternative CPU and GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    