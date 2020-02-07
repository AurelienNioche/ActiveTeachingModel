# ActiveTeachingModel

Note: All commands are given based on the assumption of using the Homebrew's package manager (MacOs).
Small changes are expected under Linux and/or Windows.

## Dependencies

#### Python 3

Using brew on mac:

    brew install python3
    
* Create a virtual environment (optional)

        source venv/bin/activate
    
    (If using venv, replace 'pip3' by pip)

* Python libraries

        pip3 install -r requirements.txt
        
#### Docker


Using Docker desktop:

    https://hub.docker.com/editions/community/docker-ce-desktop-mac
    
    
* Basic operations
 
 
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
 
 
 ## Config 
 
 #### Create db inside the container
 
1. Connect to the container

        docker-compose exec db bash
 
2. Switch to postgres user
 
        su - postgres
    
3. Connect to the db postgres (using the user postgres):

        psql
 
4. Change the password:

        ALTER USER postgres WITH password 'postgres';

5. Create db
    
        createdb ActiveTeachingModel --user=postgres
        
#### Django migrations

    python3 makemigrations
    python3 migrate

        
## Run

    python3 main_grid_dj_bkp.py
