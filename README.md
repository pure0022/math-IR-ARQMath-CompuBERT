# math-IR-ARQMath-CompuBERT
All contents of this project are for a John's Hopkins University Final Project in the course: Information Retrieval.


### Edit as of: 12/4/2022

**Credit to CompuBERT for starter code (https://github.com/MIR-MU/CompuBERT/)**
<br>
#### Training Datasets
Please download the data from the ARQMath Drive (https://drive.google.com/drive/folders/1YekTVvfmYKZ8I5uiUMbs21G2mKwF9IAm)
<br><br>
The Dataset should be downloaded and named in a folder called data (link above). To effectively process that data remove the Version number from the file. For example, *Post.V1.3.xml* should be changed to *Post.xml*. 
<br><br>
To access pickled data objects, please download from this drive (https://drive.google.com/drive/folders/1HSeqKKZc9vlM_MN_ZE25ukWLZAgPPRAg?usp=sharing)
<br>

#### Requirements
First, make sure pytorch version 1.13.0 with cuda 11.6 is enabled on your linux machine. This can be installed from the pytorch website (https://pytorch.org/get-started/locally/)<br>
Please install the requirements in the requirements.txt file. There are additional requirements to download that can be done directly in the Python Notebook provided.
<br><br>

#### The Final IR Model
The model is fine-tuned in the FinalProject.ipynb file in this GitHub. Within that file you will see every step to recreate the IR system. A sample tsv of results are provided - query-A.301.tsv. This is the results for query-A.301 from the ARQMath competition. Full Results are published in written report.
<br>
#### Evaluation
To evaluate the IR system component of the model, you must download the list of Topics for ARQMath Task 1 (download here: https://drive.google.com/drive/folders/16YHs8kqWRedSTOMDJSzHjGPg4sCl6GEC) into the same data folder where you stored the Post data. For evaluation you also need to make sure you download the latest trec_eval tool (https://trec.nist.gov/trec_eval/). Download and unzip this file in the working directory and change argument for "trec_eval" to the associated directory.
