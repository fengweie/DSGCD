
# Generalized Category Discovery under Domain Shift: A Frequency Domain Perspective
Official code for  [Generalized Category Discovery under Domain Shift: A Frequency Domain Perspective].

Generalized Category Discovery (GCD) aims to leverage labeled samples from known categories to cluster unlabeled data that may include both known and unknown categories. While existing methods have achieved impressive results under standard conditions, their performance often deteriorates in the presence of distribution shifts. In this paper, we explore a more realistic task: Domain-Shifted Generalized Category Discovery (DS_GCD), where the unlabeled data includes not only unknown categories but also samples from unknown domains. To tackle this challenge, we propose a \textbf{\underline{F}}requency-guided Gene\textbf{\underline{r}}alized Cat\textbf{\underline{e}}gory Discov\textbf{\underline{e}}ry framework (Free) that enhances the model's ability to discover categories under distributional shift by leveraging frequency-domain information. Specifically, we first propose a frequency-based domain separation strategy that partitions samples into known and unknown domains by measuring their amplitude differences. We then propose two types of frequency-domain perturbation strategies: a cross-domain strategy, which adapts to new distributions by exchanging amplitude components across domains, and an intra-domain strategy, which enhances robustness to intra-domain variations within the unknown domain. Furthermore, we extend the self-supervised contrastive objective and semantic clustering loss to better guide the training process. Finally, we introduce a clustering-difficulty-aware resampling technique to adaptively focus on harder-to-cluster categories, further enhancing model performance. Extensive experiments demonstrate that our method effectively mitigates the impact of distributional shifts across various benchmark datasets and achieves superior performance in discovering both known and unknown categories.


# Note
We will release our code soon.


## Prerequisite 🛠️

First, you need to clone the HiLo repository from GitHub. Open your terminal and run the following command:

```
git clone https://github.com/Visual-AI/free.git
cd free
```

We recommend setting up a conda environment for the project:

```bash
conda create --name=free python=3.9
conda activate free
pip install -r requirements.txt
```

## Running 🏃
### Config

Set paths to datasets and desired log directories in ```config.py```


### Datasets

We use DomainNet and our created Semantic Shift Benchmark Corruption (SSB-C) datasets:

* [DomainNet](https://ai.bu.edu/M3SDA/)
* SSB-C ([Personal OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/hjwang_connect_hku_hk/EeeL3WQ0zWdEsXhmmYeHBUUBnNVpNWbVm7mVA-jiyNVnNw?e=Dc4pWl) / [HKU Data Repository](https://doi.org/10.25442/hku.28607261))

### Scripts

**Train the model**:

```
bash scripts/free/domainnet.sh 0
bash scripts/free/ssbc.sh 0
```
Just be aware to make necessary changes (e.g., ``PYTHON``, ``SAVE_DIR``, ``WEIGHTS_PATH``, etc).



## Citing this work
<span id="jump"></span>
If you find this repo useful for your research, please consider citing our paper.
