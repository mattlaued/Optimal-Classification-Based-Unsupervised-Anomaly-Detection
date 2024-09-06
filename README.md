# Optimal Classification-based Anomaly Detection with Neural Networks: Theory and Practice

This contains the code for our theoretically-grounded neural network for anomaly detection.
To reproduce our results, follow the steps below:
1. Download required packages `pip install -r requirements.txt`
2. Download the corresponding datasets
(either [NSL-KDD](https://web.archive.org/web/20150205070216/http://nsl.cs.unb.ca/NSL-KDD/) [1] 
KDDTrain+.TXT and KDDTest+.TXT or
[Kitsune](https://github.com/ymirsky/Kitsune-py) [2] zipped dataset folders) and change the `dataset_name` in run.py.
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