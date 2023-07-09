# A Winning Approach: Effective Training Tricks for Nuclear Segmentation, Classification, and Composition

This repository contains PyTorch code for the paper:
KA Winning Approach: Effective Training Tricks for Nuclear Segmentation, Classification, and Composition

This paper presents a practical and innovative approach for nucleus recognition and cellular composition analysis. Our method utilizes a standard U-Net model but focuses on effective training and testing strategies instead of designing new models. The model outputs three maps where each pixel determines its nucleus and category independently, overcoming the challenge of overlapping nuclei. To achieve robust and generalizable results, we introduce practical model training tricks and model ensemble methods. These significantly enhance performance, reducing overfitting and improving applicability to new data. Our approach also addresses the scarcity of large annotated datasets by using diverse training data from five sources.


![](images/pipeline.png)

Links to the checkpoints can be found in the inference description below.



## Overlaid Classification Prediction

<p float="left">
  <img src="images/results_vis.png" alt="Segmentation" width="870" />
</p>

Results of different classification methods on histopathological patches of 20x in Lizard



## Acknowledgement
Our code on the byol branch is modified from [hovernet](https://github.com/vqdang/hover_net).



## Citation

If any part of this code is used, please give appropriate citation to our paper. <br />

```
@ARTICLE{9869632,
  author={Zhang, Wenhua and Zhang, Jun and Yang, Sen and Wang, Xiyue and Yang, Wei and Huang, Junzhou and Wang, Wenping and Han, Xiao},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Knowledge-Based Representation Learning for Nucleus Instance Classification from Histopathological Images}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2022.3201981}}
```

## Authors

* [Wenhua Zhang](https://github.com/WinnieLaugh)

## License

The dataset provided here is for research purposes only. Commercial use is not allowed. The data is held under the following license:
[Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)


