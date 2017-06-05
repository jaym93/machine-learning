##Supervised Learning: Algorithm Implementations

The source code for this project can be found alongside this readme. 
Inside is a requirements.txt. Python 3.4 or greater is required to run the code in the files and to use the dependencies. 
Install the dependencies from `requirements.txt`:

```
pip install -r requirements.txt
```

###Datasets
In `supervised` directory:
- The datasets are in the `data` folder, under respective directory names
- They been cleaned as per `cleaning_process.txt`
- The original automotive data was taken from [here](https://archive.ics.uci.edu/ml/datasets/automobile) and is in `data/auto/auto.csv`
- The original occupancy dataset was taken from [here](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+) and is distributed in `data/occupancy/datatrain.csv` and `data/occupancy/datatest.csv`
- Cleaned dataset is in `*_cleaned.csv`

###Running the algorithms
Every **.py** file within `src` can be run directly with 
```
python3 <name.py>
```
with the exception of **auto_data.py** and **occupancy_data.py** which are modules imported for loading data. 

There are two sections in auto_data and occupancy_data - **ONE OF THEM NEEDS TO BE COMMENTED OUT**
- use the first one (with Pandas frame import) for visualizing classification/regression
- use COMPLETE DATA (with the CSV import) for complete dataset classification/regression

The variables within each file are documented at [scikit learn's website](http://scikit-learn.org/stable/index.html) and may be changed at will. Just make sure the code runs again. 
Create a folder structure (at top level) as below to provide a place to record graphs / data
- images
    - boosting
        - auto
        - occupancy
    - boosting
        - auto
        - occupancy
    - boosting
        - auto
        - occupancy
    - boosting
        - auto
        - occupancy
    - boosting
        - auto
        - occupancy
        
Upon completeion, each file should write
- data.csv: with algorithm parameters and error measures
- rawdata.txt: containing a timestamp and additional metrics like confusion matrix and scores
- images of visualization
