# Chatty
A chat bot that uses Natural Language Processing and a Neural Network to carry out basic conversations.

## Technologies
The dataset used to train the model was written using JSON. The Natural Language Processing techniques used to apply preprocessing on the data so that is able to be used to train the model are implemented using NLTK. The Neural Network is built and trained using PyTorch and NumPy. The chat graphical user interface was built with Tkinter.

## How do I chat?
Before you start, you should make sure that you have the latest version of Python, Git and pip.

### Clone the project 
At the top of the root page of the repository is a green button that says "Code". Click this button, then click "HTTPS", then click the little clipboard next to the url. This will copy the url you will need to paste next. Now go to your preferred terminal and type `git clone <paste the link just copied>` and hit enter. 

### Set up virtual environment
Change into the project directory using `cd <absolute or relative path to project directory>`. There are many virtual environments for Python, but we will use pipenv here. Install pipenv using `pip install pipenv`, then activate it using `pipenv shell`. 

### Install dependencies
Run the command `pipenv install`.

### Train the model
Run the command `pipenv run python train.py`.

### Start the app
Run the command `pipenv run python app.py`.

## What can I talk about?
Chatty is designed to carry out basic conversation, and is equipped with a collection of jokes and fun facts to share. Chatty is also able to provide help to users that need more information.
