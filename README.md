# Optimal Classification-based Anomaly Detection with Neural Networks: Theory and Practice

This contains the code for our theoretically-grounded neural network for anomaly detection.
To reproduce our results, follow the steps below:
1. Download required packages `pip install -r requirements.txt`
2. Download the corresponding datasets
(either [NSL-KDD](https://web.archive.org/web/20150205070216/http://nsl.cs.unb.ca/NSL-KDD/) [1] 
KDDTrain+.TXT and KDDTest+.TXT,
[Kitsune](https://github.com/ymirsky/Kitsune-py) [2] zipped dataset folders, 
[Thyroid](https://archive.ics.uci.edu/dataset/102/thyroid+disease) [3] or 
[MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) [4]) and change the `dataset_name` in run.py.
    1. Note that for MVTec, we recommend using pre-trained embeddings. 
   In our experiments, we used frozen embeddings from [DINOv2](https://huggingface.co/docs/transformers/en/model_doc/dinov2) [5].
   You may run the Jupyter notebook in the `mvtec` folder to obtain these embeddings (be sure to check the paths of the data).
3. Change hyperparameters in the `args` dictionary in `Utils/run.py` as desired.
4. Run `python Utils/run.py` in the terminal.
5. Watch console for averaged metrics. 
First set of averaged metrics corresponds to threshold estimated at 5% threshold, 
while second set is at the middle threshold.
We only consider AUPR, so the threshold does not matter for the most part.


[1] Mahbod Tavallaee, Ebrahim Bagheri, Wei Lu, and Ali A. Ghorbani. A detailed analysis of the KDD cup 99
data set. In 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications,
pages 1–6, 2009. doi: 10.1109/CISDA.2009.5356528.

[2] Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai. Kitsune: An ensemble of autoencoders
for online network intrusion detection. In Proceedings 2018 Network and Distributed System Security
Symposium (NDSS). Internet Society, 2018.

[3] Ross Quinlan. Thyroid Disease. UCI Machine Learning Repository, 1987. DOI:
https://doi.org/10.24432/C5D010.

[4] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Mvtec ad — a comprehensive
real-world dataset for unsupervised anomaly detection. In 2019 IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), pp. 9584–9592, 2019. doi: 10.1109/CVPR.2019.00982.

[5] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre
Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech
Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel
Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski.
Dinov2: Learning robust visual features without supervision, 2023.