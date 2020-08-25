# ActiveTeachingModel

Note: All commands are given based on the assumption of using the Homebrew's package manager (MacOs).

## Dependencies

#### Python 3

Using brew on mac:

    brew install python3

* Create a virtual environment (optional)

        source venv/bin/activate

    (If using venv, replace 'pip3' by pip)

* Python libraries

        pip3 install -r requirements.txt


 ## Triton
Create the config files

    generate_config_files_triton.py
    
Edit the number of array job in `simulation.job`
 
Launch the job with:
 
    ./run.sh simulation.job
    
See job status:
    
    sacct -u <user>
    
Count number of results files:
    
    ls data | wc -l
    
Check the last 10 lines of the log file:

    tail -f triton_out/out
    
or for seeing the complete file:
    
    cat triton_out/out
    
cancel the job:
    
    scancel <job_id>
    
 
check the resources used by the job:

    seff <job_id>
