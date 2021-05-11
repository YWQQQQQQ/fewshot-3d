# few shot point cloud classification

----------------
## environment
> Python 3.8  
> Pytorch 1.4.0  
> scikit-learn 0.24.1  
> torchvision 0.5.0  
> h5py 2.10.0

## before usage
Run git clone command to download the source code
> git clone https://github.com/YWQQQQQQ/fewshot-3d.git

Download **Modelnet40** dataset (The code will download itself, manually download and unzip to 'dataset' folder if error appears)
> https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
## usage

5-way-5-shot-5-task pointnet EGNN BCELoss
> python train.py --device cuda --num_ways 5 --num_shots 5 --num_tasks 5 --emb_net pointnet --gnn_net egnn --loss bce

5-way-5-shot-5-task pointnet ours circle Loss  
> python train.py --device cuda --num_ways 5 --num_shots 5 --num_tasks 5 --emb_net pointnet --gnn_net ours --loss circle

5-way-5-shot-5-task pointnet ours circle Loss node_feats==64
> python train.py --device cuda --num_ways 5 --num_shots 5 --num_tasks 5 --emb_net pointnet --gnn_net ours --loss circle --num_emb_feats 64

5-way-5-shot-5-task pointnet ours circle Loss  gnn_layer==8 (82.2)
> python train.py --device cuda --num_ways 5 --num_shots 5 --num_tasks 5 --emb_net pointnet --gnn_net ours --loss circle --num_graph_layers 8

5-way-5-shot-5-task dgcnn ours circle Loss
> python train.py --device cuda --num_ways 5 --num_shots 5 --num_tasks 5 --emb_net ldgcnn --gnn_net ours --loss circle