Toward Enhanced Robustness in Unsupervised Graph Representation Learning: A Graph Information Bottleneck Perspective
===============================================================================

About
-----

This project is the implementation of the paper "Toward Enhanced Robustness in Unsupervised Graph Representation
Learning: A Graph Information Bottleneck Perspectiv".

Dependencies
-----

The following packages are mainly required (along with their dependencies):

- `torch==1.12.1`
- `torch-geometric==2.3.1`

Some Python module dependencies are listed in `requirements.txt`, which can be easily installed with pip:

```
pip install -r requirements.txt
```

Usage: Model Training
-----

You can train the robust representations by the main script ```main.py```

### Main Script

The help information of the main script ```main.py``` is listed as follows:

    python main.py -h
    
    usage: main.py [-h] [--dataset DATASET] [--emb_dim EMB_DIM]
               [--nlayers NLAYERS] [--training] [--adv] [--rate RATE]
               [--eps_x EPS_X] [--alpha ALPHA] [--beta BETA] [--device DEVICE]

    Process some integers.
    
    optional arguments:
      -h, --help         show this help message and exit
      --dataset DATASET  The required dataset
      --emb_dim EMB_DIM  The embedding dim of the encoder
      --nlayers NLAYERS  The number of layers of the GNN
      --training         Whether train the encoder or load exising encoder
      --adv              Whether evaluate the representations on the adversarial
                         graph
      --rate RATE        The edge attack ratio
      --eps_x EPS_X      The features attack ratio
      --alpha ALPHA      The hyper-parameter to control the weight of the
                         adversarial term
      --beta BETA        The hyper-parameter to control the weight of the KL
                         divergence term
      --device DEVICE    The device you use

### Demo

Then a demo script is available by calling ```main.py```, as the following:

    python main.py --dataset cora --alpha 0.9 --beta 0.1 --rate 0.2 --eps_x 0.001 --training --device 1

Usage: Evaluation
-----
To evaluate the robustness on the adversarial graph, you need to generate the adversarial graphs first. We provide the
script ```generate_adv.py``` to generate the adversarial graphs by [Deeprobust](https://github.com/DSE-MSU/DeepRobust).  


The help information of the main script ```generate_adv.py``` is listed as follows:

    python generate_adv.py -h

    usage: generate_adv.py [-h] [--dataset DATASET] [--rate RATE] [--eps_x EPS_X]
                           [--device DEVICE]
    
    optional arguments:
      -h, --help         show this help message and exit
      --dataset DATASET  The dataset required.
      --rate RATE        The ratio for edge perturbations.
      --eps_x EPS_X      The ratio for feature perturbations.
      --device DEVICE    The devive you use.


### Demo

Then a demo script is available by calling ```generate_adv.py```, as the following:

    python generate_adv.py --dataset cora

Then, you can evaluate the robustness of RGIB with the adversarial graphs:

    python main.py --dataset cora --alpha 0.9 --beta 0.1 --rate 0.2 --eps_x 0.001 --adv --device 1

      
