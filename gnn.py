# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.10.4 ('base')
#     language: python
#     name: python3
# ---

# %%
import os
import uproot as ur
import numpy as np
import matplotlib.pyplot as plt
from bitstring import BitArray
import pandas as pd

# %% [markdown]
# Load Data from ROOT Files

# %%
paths = []

for (path, dirnames, filenames) in os.walk('/home/dmisra/eic/zdc_neutron_samples/'):
    paths.extend(os.path.join(path, name) for name in filenames)

# %%
samples = {}

for path in paths:
    with ur.open(path) as file:
       tree = file["events"]
       samples[os.path.basename(f'{path}')] = tree.arrays()


# %% [markdown]
# Helper Functions

# %%
def bitExtract(n, k, p):  
    return (((1 << k) - 1)  &  (n >> p))

#Extract signed integer from bitstring
def signedint(xbits):
    x_int = []
    x_bin = np.vectorize(np.binary_repr, otypes=[str])(xbits, width=12)
    for bits in x_bin:
            x_int.append(BitArray(bin=bits).int)
    return np.array(x_int)


# %%
def get_labels(data, count):
    energy_labels = []
    for i in range(count):
        label = np.sqrt(data["MCParticles.momentum.x"][0,0]**2 + data["MCParticles.momentum.y"][0,0]**2 + data["MCParticles.momentum.z"][0,0]**2)
        if len(data['ZDC_WSi_Hits.energy'][i]) > 0 or len(data['ZDC_PbSi_Hits.energy'][i]) > 0:
            energy_labels.append(label)

    return energy_labels


# %%
def get_layerIDs(data, branch, events):
    layerID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)
        if len(data['ZDC_WSi_Hits.energy'][i]) + len(data['ZDC_PbSi_Hits.energy'][i]) > 0:
            layerID.append(event_layerID)
            
    return layerID


# %%
def get_eDep(data, branch, events):
    hitsEnergy = []
    for i in range(events):
        event_hitsEnergy = np.array(data[f"{branch}.energy"][i])
        if len(data['ZDC_WSi_Hits.energy'][i]) + len(data['ZDC_PbSi_Hits.energy'][i]) > 0:
            hitsEnergy.append(event_hitsEnergy)

    return hitsEnergy


# %%
def get_xIDs(data, branch, events):
    xID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_xID = signedint(bitExtract(event_cellID, 12, 24))
        if len(event_cellID) > 0:
            xID.append(event_xID)
        elif len(data['ZDC_WSi_Hits.energy'][i]) > 0:
            xID.append(np.array([]))
            
    return xID


# %%
def get_xIDs_WSi(data, branch, events):
    xID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)
        event_xID = signedint(bitExtract(event_cellID, 12, 24))
        for i in range(len(event_layerID)):
            if event_layerID[i] in [1, 12, 23]:
                event_xID[i] = 0.3 * event_xID[i]
        if len(event_cellID) > 0:
            xID.append(event_xID)
        elif len(data['ZDC_PbSi_Hits.energy'][i]) > 0:
            xID.append(np.array([]))
            
    return xID


# %%
def get_yIDs(data, branch, events):
    yID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_yID = signedint(bitExtract(event_cellID, 12, 36))
        if len(event_cellID) > 0:
            yID.append(event_yID)
        elif len(data['ZDC_WSi_Hits.energy'][i]) > 0:
            yID.append(np.array([]))
            
    return yID


# %%
def get_yIDs_WSi(data, branch, events):
    yID = []
    for i in range(events):
        event_cellID = np.array(data[f"{branch}.cellID"][i])
        event_layerID = bitExtract(event_cellID, 6, 8)
        event_yID = signedint(bitExtract(event_cellID, 12, 36))
        for i in range(len(event_layerID)):
            if event_layerID[i] in [1, 12, 23]:
                event_yID[i] = 0.3 * event_yID[i]
        if len(event_cellID) > 0:
            yID.append(event_yID)
        elif len(data['ZDC_PbSi_Hits.energy'][i]) > 0:
            yID.append(np.array([]))

    return yID


# %% [markdown]
# Get Features and Labels

# %%
import pickle

# %%
nevents = 10000

# %%
hitEnergyDep = dict()
xIDs = dict()
yIDs = dict()
layerIDs = dict()

# %%
eDep_WSi_10 = get_eDep(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
eDep_PbSi_10 = get_eDep(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
xIDs_WSi_10 = get_xIDs_WSi(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
xIDs_PbSi_10 = get_xIDs(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
yIDs_WSi_10 = get_yIDs_WSi(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
yIDs_PbSi_10 = get_yIDs(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
layerIDs_WSi_10 = get_layerIDs(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
layerIDs_PbSi_10 = [x + 23 for x in get_layerIDs(samples["zdc_neutron_10GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)]

# %%
hitEnergyDep['10GeV'] = [np.concatenate([eDep_WSi_10[i], eDep_PbSi_10[i]]) for i in range(len(eDep_WSi_10))]
xIDs['10GeV'] = [np.concatenate([xIDs_WSi_10[i], xIDs_PbSi_10[i]]) for i in range(len(eDep_WSi_10))]
yIDs['10GeV'] = [np.concatenate([yIDs_WSi_10[i], yIDs_PbSi_10[i]]) for i in range(len(eDep_WSi_10))]
layerIDs['10GeV'] = [np.concatenate([layerIDs_WSi_10[i], layerIDs_PbSi_10[i]]) for i in range(len(eDep_WSi_10))]

with open('hitEnergyDep_10GeV', 'wb') as handle:
    pickle.dump(hitEnergyDep, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('xIDs_10GeV', 'wb') as handle:
    pickle.dump(xIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('yIDs_10GeV', 'wb') as handle:
    pickle.dump(yIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('layerIDs_10GeV', 'wb') as handle:
    pickle.dump(layerIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
eDep_WSi_20 = get_eDep(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
eDep_PbSi_20 = get_eDep(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
xIDs_WSi_20 = get_xIDs_WSi(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
xIDs_PbSi_20 = get_xIDs(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
yIDs_WSi_20 = get_yIDs_WSi(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
yIDs_PbSi_20 = get_yIDs(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
layerIDs_WSi_20 = get_layerIDs(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
layerIDs_PbSi_20 = [x + 23 for x in get_layerIDs(samples["zdc_neutron_20GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)]

# %%
hitEnergyDep['20GeV'] = [np.concatenate([eDep_WSi_20[i], eDep_PbSi_20[i]]) for i in range(len(eDep_WSi_20))]
xIDs['20GeV'] = [np.concatenate([xIDs_WSi_20[i], xIDs_PbSi_20[i]]) for i in range(len(eDep_WSi_20))]
yIDs['20GeV'] = [np.concatenate([yIDs_WSi_20[i], yIDs_PbSi_20[i]]) for i in range(len(eDep_WSi_20))]
layerIDs['20GeV'] = [np.concatenate([layerIDs_WSi_20[i], layerIDs_PbSi_20[i]]) for i in range(len(eDep_WSi_20))]

with open('hitEnergyDep_20GeV', 'wb') as handle:
    pickle.dump(hitEnergyDep, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('xIDs_20GeV', 'wb') as handle:
    pickle.dump(xIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('yIDs_20GeV', 'wb') as handle:
    pickle.dump(yIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('layerIDs_20GeV', 'wb') as handle:
    pickle.dump(layerIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
eDep_WSi_50 = get_eDep(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
eDep_PbSi_50 = get_eDep(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
xIDs_WSi_50 = get_xIDs_WSi(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
xIDs_PbSi_50 = get_xIDs(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
yIDs_WSi_50 = get_yIDs_WSi(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
yIDs_PbSi_50 = get_yIDs(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
layerIDs_WSi_50 = get_layerIDs(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
layerIDs_PbSi_50 = [x + 23 for x in get_layerIDs(samples["zdc_neutron_50GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)]

# %%
hitEnergyDep['50GeV'] = [np.concatenate([eDep_WSi_50[i], eDep_PbSi_50[i]]) for i in range(len(eDep_WSi_50))]
xIDs['50GeV'] = [np.concatenate([xIDs_WSi_50[i], xIDs_PbSi_50[i]]) for i in range(len(eDep_WSi_50))]
yIDs['50GeV'] = [np.concatenate([yIDs_WSi_50[i], yIDs_PbSi_50[i]]) for i in range(len(eDep_WSi_50))]
layerIDs['50GeV'] = [np.concatenate([layerIDs_WSi_50[i], layerIDs_PbSi_50[i]]) for i in range(len(eDep_WSi_50))]

with open('hitEnergyDep_50GeV', 'wb') as handle:
    pickle.dump(hitEnergyDep, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('xIDs_50GeV', 'wb') as handle:
    pickle.dump(xIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('yIDs_50GeV', 'wb') as handle:
    pickle.dump(yIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('layerIDs_50GeV', 'wb') as handle:
    pickle.dump(layerIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
eDep_WSi_100 = get_eDep(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
eDep_PbSi_100 = get_eDep(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
xIDs_WSi_100 = get_xIDs_WSi(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
xIDs_PbSi_100 = get_xIDs(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
yIDs_WSi_100 = get_yIDs_WSi(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
yIDs_PbSi_100 = get_yIDs(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
layerIDs_WSi_100 = get_layerIDs(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
layerIDs_PbSi_100 = [x + 23 for x in get_layerIDs(samples["zdc_neutron_100GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)]

# %%
hitEnergyDep['100GeV'] = [np.concatenate([eDep_WSi_100[i], eDep_PbSi_100[i]]) for i in range(len(eDep_WSi_100))]
xIDs['100GeV'] = [np.concatenate([xIDs_WSi_100[i], xIDs_PbSi_100[i]]) for i in range(len(eDep_WSi_100))]
yIDs['100GeV'] = [np.concatenate([yIDs_WSi_100[i], yIDs_PbSi_100[i]]) for i in range(len(eDep_WSi_100))]
layerIDs['100GeV'] = [np.concatenate([layerIDs_WSi_100[i], layerIDs_PbSi_100[i]]) for i in range(len(eDep_WSi_100))]

with open('hitEnergyDep_100GeV', 'wb') as handle:
    pickle.dump(hitEnergyDep, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('xIDs_100GeV', 'wb') as handle:
    pickle.dump(xIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('yIDs_100GeV', 'wb') as handle:
    pickle.dump(yIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('layerIDs_100GeV', 'wb') as handle:
    pickle.dump(layerIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
eDep_WSi_150 = get_eDep(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
eDep_PbSi_150 = get_eDep(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
xIDs_WSi_150 = get_xIDs_WSi(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
xIDs_PbSi_150 = get_xIDs(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
yIDs_WSi_150 = get_yIDs_WSi(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
yIDs_PbSi_150 = get_yIDs(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)
layerIDs_WSi_150 = get_layerIDs(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_WSi_Hits", nevents)
layerIDs_PbSi_150 = [x + 23 for x in get_layerIDs(samples["zdc_neutron_150GeV_10e4.edm4hep.root"], "ZDC_PbSi_Hits", nevents)]

# %%
hitEnergyDep['150GeV'] = [np.concatenate([eDep_WSi_150[i], eDep_PbSi_150[i]]) for i in range(len(eDep_WSi_150))]
xIDs['150GeV'] = [np.concatenate([xIDs_WSi_150[i], xIDs_PbSi_150[i]]) for i in range(len(eDep_WSi_150))]
yIDs['150GeV'] = [np.concatenate([yIDs_WSi_150[i], yIDs_PbSi_150[i]]) for i in range(len(eDep_WSi_150))]
layerIDs['150GeV'] = [np.concatenate([layerIDs_WSi_150[i], layerIDs_PbSi_150[i]]) for i in range(len(eDep_WSi_150))]

with open('hitEnergyDep_150GeV', 'wb') as handle:
    pickle.dump(hitEnergyDep, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('xIDs_150GeV', 'wb') as handle:
    pickle.dump(xIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('yIDs_150GeV', 'wb') as handle:
    pickle.dump(yIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('layerIDs_150GeV', 'wb') as handle:
    pickle.dump(layerIDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
with open('hitEnergyDep_150GeV', 'rb') as handle:
    hitEnergyDep = pickle.load(handle)
with open('xIDs_150GeV', 'rb') as handle:
    xIDs = pickle.load(handle)
with open('yIDs_150GeV', 'rb') as handle:
    yIDs = pickle.load(handle)
with open('layerIDs_150GeV', 'rb') as handle:
    layerIDs = pickle.load(handle)

# %% [markdown]
# Merge Data

# %%
import awkward as ak
from sklearn.model_selection import train_test_split

# %%
data_labels = np.concatenate([get_labels(samples[key], nevents) for key in samples])

# %%
hitEnergyDep_all = ak.concatenate(list(hitEnergyDep.values()), axis=0)
xIDs_all = ak.concatenate(list(xIDs.values()), axis=0)
yIDs_all = ak.concatenate(list(yIDs.values()), axis=0)
layerIDs_all = ak.concatenate(list(layerIDs.values()), axis=0)

# %%
hitEnergyDep_train, hitEnergyDep_test, xIDs_train, xIDs_test, yIDs_train, yIDs_test, layerIDs_train, layerIDs_test, labels_train, labels_test = train_test_split(hitEnergyDep_all, xIDs_all, yIDs_all, layerIDs_all, data_labels, test_size=0.2, train_size=0.8, random_state=None, shuffle=True)

# %%
features_train = [hitEnergyDep_train, xIDs_train, yIDs_train, layerIDs_train]
features_test = [hitEnergyDep_test, xIDs_test, yIDs_test, layerIDs_test]

# %% [markdown]
# PyTorch Geometric

# %%
import torch
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph, GCNConv, global_add_pool


# %%
class ZDCDataset(Dataset):
    def __init__(self, features, labels, knn_k=4):
        super(ZDCDataset, self).__init__()
        
        self.knn_k = knn_k
        
        self.energy = features[0]
        self.xID = features[1]
        self.yID = features[2]
        self.layerID = features[3]
        self.label = labels

    def len(self):
        return len(self.energy)

    def get(self, idx):
        
        energy = torch.tensor(self.energy[idx]).to(torch.float32)
        xID = torch.tensor(self.xID[idx]).to(torch.float32)
        yID = torch.tensor(self.yID[idx]).to(torch.float32)
        layerID = torch.tensor(self.layerID[idx]).to(torch.float32)
        
        label = torch.tensor(self.label[idx]).to(torch.float32)
        
        x = torch.stack([energy, xID, yID, layerID], axis=-1)
        
        #construct knn graph from (x, y, z) coordinates
        edge_index = knn_graph(x[:, [1,3]], k=self.knn_k, num_workers=32)
        
        data = Data(
            x = x,
            y = label,
            edge_index = edge_index
        )
        
        return data


# %%
dataset = ZDCDataset(features_train, labels_train, knn_k=4)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers = 32)

# %%
ibatch = 0
for data_batched in loader:
    print(ibatch, data_batched.x.shape, data_batched.y)
    ibatch += 1
    if ibatch>5:
        break

# %% [markdown]
# GCN Model

# %%
from torch_geometric.nn import GCNConv, global_add_pool

class Net(torch.nn.Module):
    def __init__(self, num_node_features=4):
        super(Net, self).__init__()
        
        #(4 -> N)
        self.conv1 = GCNConv(num_node_features, 256)
        
        #(N -> 1)
        self.output = torch.nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        #add a batch index, in in case we are running on a single graph
        if not hasattr(data, "batch"):
            data.batch = torch.zeros(len(x), dtype=torch.int64).to(x.device)
        
        #Transform the nodes with the graph convolution
        transformed_nodes = self.conv1(x, edge_index)
        transformed_nodes = torch.nn.functional.elu(transformed_nodes)
        
        #Sum up all the node vectors in each graph according to the batch index
        per_graph_aggregation = global_add_pool(transformed_nodes, data.batch)
        
        #For each graph,
        #predict the output based on the total vector
        #from the previous aggregation step
        output = self.output(per_graph_aggregation)
        return output


# %%
net = Net()

# %%
net(data_batched)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#scheduler = ExponentialLR(optimizer, gamma=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# %%
torch.set_num_threads(32)

# %%
model.train()
losses_train = []

for epoch in range(1000):
    
    loss_train_epoch = []
    
    for data_batch in loader:
        
        data_batch = data_batch.to(device)
        optimizer.zero_grad()
        out = model(data_batch)
        loss = nn.functional.mse_loss(out[:, 0], data_batch.y)
        loss.backward()
        loss_train_epoch.append(loss.item())
        optimizer.step()

    loss_train_epoch = np.mean(loss_train_epoch)
    losses_train.append(loss_train_epoch)
    print(epoch, loss_train_epoch)

    scheduler.step(loss_train_epoch)

    if epoch % 100 == 0:
        torch.save(obj=model.state_dict(), f=f"/home/dmisra/eic/gnn_state_dict_{epoch}")

# %%
#model.load_state_dict(torch.load('/home/dmisra/eic/gnn_state_dict_300'))

# %%
plt.plot(losses_train, label="training")
plt.ylabel("Loss")
plt.xlabel("epoch")

# %%
ievent = 42
data = dataset.get(ievent).to(device)
embedded_nodes = model.conv1(data.x, data.edge_index)

# %%
test_dataset = ZDCDataset(features_test, labels_test, knn_k=4)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers = 32)

# %%
next(iter(test_loader))

# %% [markdown]
# Predictions

# %%
from scipy.stats import norm
from scipy.optimize import curve_fit

# %%
#Set the model in evaluation mode
model.eval()

#Setup the inference mode context manager
with torch.inference_mode():
  y_preds = model(next(iter(test_loader)))

plt.hist(y_preds[:,0].numpy(),100,histtype='step')
plt.xlabel('Energy (GeV)')
plt.ylabel('Count')
plt.title('Predicted Energy Distribution')


# %%
def tensorIntersect(t1, t2):
    a = set((tuple(i) for i in t1.numpy()))
    b = set((tuple(i) for i in t2.numpy()))
    c = a.intersection(b)
    tensorform = torch.from_numpy(np.array(list(c)))

    return tensorform


# %%
#Set the model in evaluation mode
model.eval()

#Setup the inference mode context manager
with torch.inference_mode():
  y_preds_200GeV = model(features_150GeV)
  y_preds_100GeV = model(features_100GeV)
  y_preds_50GeV = model(features_50GeV)
  y_preds_20GeV = model(features_20GeV)
  y_preds_10GeV = model(features_10GeV)

# %%
peak_preds = norm.fit(y_preds_10GeV)[0], norm.fit(y_preds_20GeV)[0], norm.fit(y_preds_50GeV)[0], norm.fit(y_preds_100GeV)[0], norm.fit(y_preds_150GeV)[0]
true_peaks = [10,20,50,100,150]
peak_preds

# %%
plt.scatter(true_peaks,peak_preds)
plt.xlabel('Particle Energy (GeV)')
plt.ylabel('Reconstructed Energy (GeV)')
plt.plot(np.arange(1,201),np.arange(1,201))
plt.title('Linearity')


# %%
#Get energy resolution from distribution of predictions
def res(preds,energy):
    return norm.fit(preds)[1]/energy

energy_list = [200,100,50,10]
resolutions = res(y_preds_200GeV,200), res(y_preds_100GeV,100), res(y_preds_50GeV,50), res(y_preds_10GeV,10)


# %%
#Curve fit for energy resolution as a function of energy
def f(E,a):
    return a/np.sqrt(E)

popt, pcov = curve_fit(f, energy_list, resolutions)

# %%
popt, pcov

# %%
plt.plot(range(200),f(range(1,201),popt[0]))
plt.scatter(energy_list,resolutions)
plt.xlabel('Energy (GeV)')
plt.ylabel('Resolution')
plt.title('Energy Resolution')

# %%
torch.save(obj=model_1.state_dict(), f="/home/dmisra/eic/model_1")

# %%
