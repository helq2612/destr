# DESTR: Object Detection with Split Transformer (CVPR 2022)

This repository is an official implementation of the CVPR 2022 paper "[DESTR: Object Detection with Split Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/He_DESTR_Object_Detection_With_Split_Transformer_CVPR_2022_paper.pdf)". 

## Introduction

Contributions:
1. Split estimation of cross attention into two independent branches: one tailored for classification and the other for box regression;
2. Insert a mini-detector between encoder and decoder to initialize objects’ classification, regression and positional embeddings;
3. Augment self-attention in decoder to pair self-attention for every two pairs of spatially adjacent queries to improve inductive bias.

## Model Zoo

We provide conditional DETR and conditional DETR-DC5 models.
AP is computed on COCO 2017 *val*.

<table>
  <thead>
    <tr style="text-align: right;">
      <th>Method</th>
      <th>Epochs</th>
      <th>Params (M)</th>
      <th>AP</th>
      <th>AP<sub>S</sub></th>
      <th>AP<sub>M</sub></th>
      <th>AP<sub>L</sub></th>
      <th>URL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DETR-R50</td>
      <td>500</td>
      <td>41</td>
      <td>42.0</td>
      <td>20.5</td>
      <td>45.8</td>
      <td>61.1</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">model</a> <br/> <a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50_log.txt">log</a></td>
    </tr>
    <tr>
      <td>DETR-R50</td>
      <td>50</td>
      <td>41</td>
      <td>34.8</td>
      <td>13.9</td>
      <td>37.3</td>
      <td>54.4</td>
      <td><a href="https://pkueducn-my.sharepoint.com/:u:/g/personal/pkucxk_pku_edu_cn/EWH1OGLL4N5DufAj4ncYuigB8q62uEw6I10G-d7OD7q09A?e=SVbj7O">model</a> <br/> <a href="https://pkueducn-my.sharepoint.com/:t:/g/personal/pkucxk_pku_edu_cn/EdkDyyp9TvZIh813TR-Z4oYBYb-BfjHrsH66vqiX4IvdFA?e=SPSaec">log</a></td>
    </tr>
    <tr>
      <td><b>Conditional DETR-R50</b></td>
      <td>50</td>
      <td>44</td>
      <td>41.0</td>
      <td>20.6</td>
      <td>44.3</td>
      <td>59.3</td>
      <td><a href="https://pkueducn-my.sharepoint.com/:u:/g/personal/pkucxk_pku_edu_cn/EXaUwp6Qt29Mub0bVOExLusBlyqqyO7qCIQfVWclbOGulw?e=IrJ1Sg">model</a> <br/> <a href="https://pkueducn-my.sharepoint.com/:t:/g/personal/pkucxk_pku_edu_cn/EZR-UQF8kB5Nl0V2ojr4QgwBjTOVYcxfGrRLbQwuw-2rYA?e=Bfd6i6">log</a></td>
    </tr>
    <tr>
      <td><b>DESTR-R50</b></td>
      <td>50</td>
      <td>69</td>
      <td>43.6</td>
      <td>23.5</td>
      <td>47.6</td>
      <td>62.4</td>
      <td><a href="https://drive.google.com/file/d/1B-dZyu0a1F4Q20whfVz8orWBqj9XNX_1/view?usp=sharing">model</a> <br/> <a href="https://drive.google.com/file/d/1kWkwIplM3NjVXLWKP-zERLOLWOVeiVUJ/view?usp=sharing">log</a></td>
    </tr>
  </tbody>
</table>

Note: 
1. The numbers in the table are slightly differently
   from the numbers in the paper. We re-ran some experiments when releasing the codes.
2. More weights will be release in future



## Installation, Requirement, and Usage
Please see <a href="https://github.com/Atten4Vis/ConditionalDETR">Conditional DETR </a>

## License

DESTR is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

## Aknowledgement
DESTR is build on <a href="https://github.com/Atten4Vis/ConditionalDETR">Conditional DETR </a>. We appreciate the contributions from them!


## Citation

```bibtex
@inproceedings{he2022destr,
  title={DESTR: Object Detection with Split Transformer},
  author={He, Liqiang and Todorovic, Sinisa},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9377--9386},
  year={2022}
}
```
