# molecularAtt

A pytorch implement of Drug3D-Net[<sup>1</sup>](#refer-anchor)[[code]]((https://github.com/anny0316/Drug3D-Net)) with QM9 [<sup>2,3</sup>](#refer-anchor) dataset.  
And design my network to adapt the voxel input.  
master branch - stable version  
lab branch - implement version on lab server

## todo list
- [x] TODO: load data from qm9 dataset and convert into voxel  
- [x] TODO: implement the network structure with pytorch  
- [ ] TODO: rewrite the output of the network, train one property at one time  
- [ ] TODO: adapt the epoch and batch size for a better performance  
- [ ] TODO: add Swin Transformer[<sup>4</sup>](#refer-anchor) in the network  
- [ ] TODO: design my network to rationally combine the two attention modules  

## Reference

<div id="refer-anchor"></div>

[1]Li C, Wang J, Niu Z, et al. A spatial-temporal gated attention module for molecular property prediction based on molecular geometry[J]. Briefings in Bioinformatics, 2021.[[pdf]](https://www.researchgate.net/profile/Jianmin-Wang-3/publication/350706579_A_spatial-temporal_gated_attention_module_for_molecular_property_prediction_based_on_molecular_geometry/links/60726d2b299bf1c911c1fef7/A-spatial-temporal-gated-attention-module-for-molecular-property-prediction-based-on-molecular-geometry.pdf)  
[2]Ruddigkeit L, Van Deursen R, Blum L C, et al. Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17[J]. Journal of chemical information and modeling, 2012, 52(11): 2864-2875.[[pdf]](https://pubs.acs.org/doi/pdf/10.1021/ci300415d)  
[3]Ramakrishnan R, Dral P O, Rupp M, et al. Quantum chemistry structures and properties of 134 kilo molecules[J]. Scientific data, 2014, 1(1): 1-7.[[pdf]](https://www.nature.com/articles/sdata201422)  
[4]Liu Z, Lin Y, Cao Y, et al. Swin transformer: Hierarchical vision transformer using shifted windows[J]. arXiv preprint arXiv:2103.14030, 2021.[[pdf]](https://arxiv.org/pdf/2103.14030)