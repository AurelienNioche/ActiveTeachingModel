# ActiveTeachingModel

Note: All commands are given based on the assumption of using the Homebrew's package manager (MacOs).
Small changes are expected under Linux and/or Windows.

## Configuration

#### Python 3

    brew install python3

#### Python libraries

* numpy, matplotlib, scipy

    
    pip3 install numpy matplotlib scipy
        
## Run

    python3 adaptive_teaching.py
    
    
 ## Docker
 
 
Launch the docker container
 
    docker-compose up # Can add '-d'
    

Connect to the container

    docker-compose exec db bash


Diagnose (is the server running? and so on)

    docker-compose ps  # Show processes
    docker-compose logs -f  # If running is the background

Stop 
    
    docker-compose stop
    
Remove container (be cautious while using it!)

    docker-compose down
 