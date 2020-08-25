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
 
Launch the job with:
 
    ./run.sh simulation.job

