# ActiveTeachingModel

Note: All commands are given based on the assumption of using the Homebrew's package manager (MacOs).

## Dependencies

#### Python 3

* Install Python3


    brew install python3

* Create a virtual environment


    sudo apt-get install virtualenv
    cd /var/www/html/ActiveTeachingServer
    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt


 ## Triton
Create the config files

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
    
   
   
## Reproduce figures
    
    python make_fig.py
