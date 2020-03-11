# Data

All datasets are saved here.

### MS COCO
Download the MS COCO captioning (2015) dataset from:
http://cocodataset.org/#download

#### Karpathy split
Get the karpathy split .json files from: 
https://cs.stanford.edu/people/karpathy/deepimagesent/

## Data is immutable
Don't ever edit your raw data, especially not manually, and especially not in Excel. 
Don't overwrite your raw data. Don't save multiple versions of the raw data. 
Treat the data (and its format) as immutable. The code you write should
 move the raw data through a pipeline to your final analysis. 
 You shouldn't have to run all of the steps every time you want to 
 make a new figure (see Analysis is a DAG), but anyone should be able to 
 reproduce the final products with only the code in src and the data in data/raw.

Also, if data is immutable, it doesn't need source control in the same 
way that code does. Therefore, by default, the data folder is included 
in the .gitignore file. If you have a small amount of data that rarely 
changes, you may want to include the data in the repository. Github 
currently warns if files are over 50MB and rejects files over 100MB. 
Some other options for storing/syncing large data include AWS S3 with 
a syncing tool (e.g., s3cmd), Git Large File Storage, Git Annex, and dat. 
Currently by default, we ask for an S3 bucket and use AWS CLI to sync 
data in the data folder with the server.

This section was copied from https://drivendata.github.io/cookiecutter-data-science/