# DeepCSI
Algorithm for radio fingerprinting of MU-MIMO Wi-Fi transmitters that leverages the beamforming feedback matrices sent by the beamformees to the beamformer to enable beamforming.

This repository contains the reference code for the article [''DeepCSI: Rethinking Wi-Fi Radio FingerprintingThrough MU-MIMO CSI Feedback Deep Learning''](https://arxiv.org/abs/).

If you find the project useful and you use this code, please cite our article:
```
@inproceedings{meneghello2022deepcsi,
  author = {Meneghello, Francesca and Rossi, Michele and Restuccia, Francesco},
  title = {{DeepCSI: Rethinking Wi-Fi Radio FingerprintingThrough MU-MIMO CSI Feedback Deep Learning}},
  booktitle = {IEEE International Conference on Distributed Computing Systems},
  year = {2022}
}
```

## How to use
Clone the repository and enter the folder with the python code:
```bash
cd <your_path>
git clone https://github.com/signetlabdei/DeepCSI
```

Download the input data from http://researchdata.cab.unipd.it/id/eprint/623 and unzip the file.
For your convenience, you can use the ```input_files``` inside this project folder to place the files but the scripts work whatever is the source folder.

The dataset contains the ```.pcap``` files collected through Wireshark using a laptop in monitor mode within an IEEE 802.11ac network. 
The experimental network consists of one access point (AP) and two stations (STAs). MU-MIMO is enabled in the downlink (DL) direction, i.e., the AP acts as beamformer while the STAs are the beamformees.
Specifically, the AP (beamformer) is the target of the fingerprinting algorithm. 
Ten different Wi-Fi modules are used subsequently as AP and the associated traces are collected. 

The MAC addresses of the stations are as follows:
STA1: ```14:59:C0:34:A2:57```
STA2: ```14:59:C0:5A:48:BE```

The MAC addresses of the 10 Wi-Fi modules are: ```04:f0:21:63:f8:XX``` where ```XX``` is the module-specific suffix and acts as an identifier for the different modules.

The code for DeepCSI is implemented in Matlab and Python.
The first part of the processing can be found in the `Matlab_code` folder inside this repository. 
The learning-based algorithm is in the `Python_code` folder.
The scripts to perform the processing are described in the following, together with the specific parameters.

### Beamforming feedback packets extraction
First, the ```.pcap``` files need to be processed to extract the packets containing the beamforming feedback matrices. 
The filters are implemented in the following two bash scripts. 
The processed files are saved into the `split` subfolders of the `input_files/dataset` and `input_files/dataset_mobility` folders. 
The suffix of each processed file corresponds to the beamformee that generated the feedback packet. 
```bash
./TSHARK_BASH
./TSHARK_BASH_mobility
```

### Beamforming feedback matrices reconstruction
The next steps entail the extraction of the beamforming feedback angles from the packets and the reconstruction of the beamforming feedback matrices from the quantized angles.
This is performed through the following script in the `Matlab_code` folder.
The output files consist of the angles, the beamforming feedback matrices, the exclusive beamforming report for MU-MIMO and the time vector. 
The files are save in subfolders of the `input_files/processed_dataset` and `input_files/processed_dataset_mobility` folders.
The processing of the two datasets needs to be done subsequently by changing the `mobility` variable in the Matlab script.
```bash
matlab csi_read_beamforming_matrix.m
```

### Beamforming feedback matrices dataset creation 
The following script in the `Python_code` folder creates the datasets of beamforming feedback matrices.
```bash
python create_dataset.py <'directory of input data'> <'ID of the beamformee considered (`57` or `BE`)'> <'ID of the beamformee position in {1, ..., 9}'> <'maximum number of samples to consider'> <'prefix to identify the data'> <'folder to save the dataset'> <'select random indices (`rand`) or subsample the data (`sampling`)'>
```
e.g., 
- python create_dataset.py ../input_files/processed_dataset/ 57 9 200 _ ./dataset/ sampling
- python create_dataset.py ../input_files/processed_dataset_mobility/ BE 11 1000 _m ./dataset_mobility/ sampling


### Train the learning algorithm for fingerprinting and test the performance
The following script allows training and testing DeepCSI on the scenarios identified in the reference paper based on the arguments passed as input.
The fingerprinting is beamformee-specific: one model has to been trained using the feedback matrices collected from each of the beamformees.
```bash
python learning.py <'directory of the beamforming feedback matrices dataset'> <'ID of the beamformee considered (`57` or `BE`)'> <'ID of the beamformee position in {1, ..., 9}'> <'name for the model to be saved'> <'number of transmitter antennas'> <'number of receiver antennas'> <'indices of the transmitter antennas to consider, comma separated'> <'indices of the receiver antennas to consider, comma separated'> <'bandwidth'> <'model type in {`convolutional`, `attention`}'> <'prefix to identify the data'> <'scenario considered in {S1, S2, S3, S4, S4_diff, S5, S6, hyper}'>
```
e.g., 
- python learning.py ./dataset/ 57 9 finger_ 3 2 0,1,2 0 80 attention _ S1
- python learning.py ./dataset/ 57 9 finger_ 3 2 0,1,2 0 80 attention_hyper_selection-128,128,128,128-7,7,7,5 _ hyper


### Test the performance of the algorithm on the beamforming feedback matrices extracted from a different beamformee
```bash
python learning_test_different_beamformee.py <'directory of the beamforming feedback matrices dataset'> <'ID of the beamformee considered (`57` or `BE`)'> <'ID of the beamformee position in {1, ..., 9}'> <'name for the model to be saved'> <'number of transmitter antennas'> <'number of receiver antennas'> <'indices of the transmitter antennas to consider, comma separated'> <'indices of the receiver antennas to consider, comma separated'> <'bandwidth'> <'model type in {`convolutional`, `attention`}'> <'prefix to identify the data'> <'scenario considered in {S1, S2, S3, S4, S4_diff, S5, S6, hyper}'>
```
e.g., python learning_test_different_beamformee.py ./dataset/ BE,57 9 finger_rev_ 3 2 0,1,2 0 80 attention _ S1


### Utilities
To plot the confusion matrices use the following script.
```bash
python learning_plots.py <'name for the model to be saved'> <'ID of the beamformee considered (`57` or `BE`)'> <'ID of the beamformee position in {1, ..., 9}'> <'number of transmitter antennas'> <'number of receiver antennas'> <'indices of the transmitter antennas to consider, comma separated'> <'indices of the receiver antennas to consider, comma separated'> <'bandwidth'> <'model type in {`convolutional`, `attention`}'> <'scenario considered in {S1, S2, S3, S4, S4_diff, S5, S6, hyper}'>
```
e.g., python learning_test_different_beamformee.py ./dataset/ BE,57 9 finger_rev_ 3 2 0,1,2 0 80 attention _ S1

The performance of the model by changing the hyperparameter can be visualized through the following script.
```bash
python plots_hyperparameters_selection.py <'ID of the beamformee considered (`57` or `BE`)'> <'name for the model to be saved'> <'model type in {`convolutional`, `attention`}'>
```

To visualize the comparison between the accuracy using different bandwidths, training positions and transmitter antennas use the following scripts.
```bash
python plots_accuracy_comparison_bandwidth.py <'ID of the beamformee considered (`57` or `BE`)'> <'name for the model to be saved'> <'model type in {`convolutional`, `attention`}'>
```
```bash
python plots_accuracy_training_positions.py <'ID of the beamformee considered (`57` or `BE`)'> <'name for the model to be saved'> <'model type in {`convolutional`, `attention`}'>
```
```bash
python plots_accuracy_TXantennas.py <'ID of the beamformee considered (`57` or `BE`)'> <'name for the model to be saved'> <'model type in {`convolutional`, `attention`}'>
```

Script to visualize the data.
```bash
python loading_visualization.py <'directory of input data'> <'ID of the beamformee considered (`57` or `BE`)'> <'ID of the beamformee position in {1, ..., 9}'> <'number of transmitter antennas'> <'number of receiver antennas'> <'prefix to identify the data'> <'folder where to save the plots'>
```
e.g., python loading_visualization.py ../input_files/processed_files/ 57 9 3 2 _ ./plots/

Script to compute the length of the files in the dataset.
```bash
python file_size.py <'directory of input data'> <'ID of the beamformee considered (`57` or `BE`)'> <'number of different positions considered'> <'prefix to identify the data'> 
```
e.g., python file_size.py ../input_files/processed_files/ 57 9 _

### Analysis with simulated data to assess the impact of the angles quantization
The following script in the `Matlab_code` folder allows creating synthetic Wi-Fi channel frequency response data and assess the impact of the quantization of the angles in the reconstruction of the beamforming feedback matrix.
```bash
matlab beamforming_matrix_analysis_quantization.m
```

## Contact
Francesca Meneghello 
meneghello@dei.unipd.it 
github.com/francescamen
