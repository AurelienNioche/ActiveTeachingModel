# ActiveTeachingModel

This repo is link to [Nioche et al. (2021)](https://dl.acm.org/doi/10.1145/3397481.3450696).

## Data 

A clean release of the dataset is accessible on [Zenodo](https://zenodo.org/record/5536917). 
It contains not only the data but also demographic information 
of the participants (age, gender, native language, other spoken languages), 
and the stimuli used (character and meaning).

## Code 

Note: All commands are given based on the assumption of using the Homebrew's package manager (MacOs).

### Dependencies

#### Python 3

* Install Python3


    brew install python3

* Create a virtual environment


    sudo apt-get install virtualenv
    cd /var/www/html/ActiveTeachingServer
    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt
    
### Local
Create the config files:

    gen_config_files.py

Run:

    python main_local.py

Data will be save under `data/triton/<trial_name>`.

For exploratory simulations (n learnt leitner):

    python explo_leitner.py

Data will be save under `data/explo_leitner/<param used>`.

 ### Triton (Aalto University Cluster)

Create the config files & run job

    gen_config_files.py
    
Data will be save under `data/triton/<trial_name>`
    
See job status
    
    sacct -u <user>
    
or 

    slurm q
    
Count number of results files
    
    ls data/triton/<trial_name> | wc -l
    
Check the last 10 lines of the log file

    tail -f triton_out/
    
or for seeing the complete file:
    
    cat triton_out/debug.out
    
cancel the job
    
    scancel <job_id>
    
 
check the resources used by the job

    seff <job_id>
    
   
   
### Reproduce figures

Unpack artificial data
    
    cd data/triton
    unzstd n_learnt_leitner.tar.zst
    tar -xvf n_learnt_leitner.tar

Run scripts
    
    python make_fig_artificial.py
    python make_fig_human.py
