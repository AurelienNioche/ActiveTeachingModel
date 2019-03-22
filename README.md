# ActiveTeachingModel


## Dependencies

### Python libraries

* numpy, matplotlib, scipy
* gensim
* keras
* django, psycopg2-binary


### PostgreSQL

Install postgresql (all commands are given considering the application running under MacOs)

    brew install postgresql
    
Run pgsql server (to have launchd start postgresql now and restart at login): 

    brew services start postgresql

OR if you don't want/need a background service:

    pg_ctl -D /usr/local/var/postgres start
    
### Pre-trained word2vec network

* Download at url: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
* Move the .bin in 

    
## Prepare database

Create user and db

    createuser postgres
    createdb ActiveTeaching --owner postgres


### Initialize Django

Prepare the DB (make migrations and migrate)

    python3 manage.py makemigrations
    python3 manage.py migrate
    
Create superuser in order to have access to admin interface

    python3 manage.py createsuperuser
    
### Import Kanji data
    
Import kanji data in db
    
    python3 prepare_db.py
    
### Import user data

Load user data
    
    python3 load_user_data.py
    
## Run scripts


Analyse experimental results

    python3 analysis.py
    

Check quality of model by fitting the model to itself

    python3 fit_the_model_itself.py 
    

### Other operations

## Run Django server
   
Use the (classic) Django command

    python3 manage.py runserver

## Manipulations of DB

Backup user data

    python3 backup_user_data.py

...can be also done by dumping data using pg commands:

    pg_dump --data-only  --table question --table user ActiveTeaching --inserts > data/user_and_question_tables.sql


Remove the db
    
    dropdb ActiveTeaching 

    
## Sources

### Kanji database
   
   Coming from Tamaoka, K., Makioka, S., Sanders, S. & Verdonschot, R.G. (2017). 
www.kanjidatabase.com: a new interactive online database for psychological and linguistic research on Japanese kanji 
and their compound words. Psychological Research, 81, 696-708.