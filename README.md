# Chatty
A chat bot that uses Natural Language Processing and a Neural Network to carry out basic conversations.

## Technologies
The Natural Language Processing techniques are implemented using NLTK. 

The chat graphical user interface was built with Tkinter.

## How do I chat?
Before you start, you should make sure that you have the latest version of Python, Git and Pip.

### Clone the project 
At the top of the root page of the repository is a green button that says "Code". Click this button, then click "HTTPS", then click the little clipboard next to the url. This will copy the url you will need to paste next. Now go to your preferred terminal and type `git clone <paste the link just copied>` and hit enter. 

### Set up virtual environment
Change into the project directory using `cd <absolute or relative path to project directory>`. There are many virtual environments for Python, but we will use pipenv here. Install pipenv using `pip install pipenv`, then activate it using `pipenv shell`. 

### Install dependencies
Run the command `pipenv install`.

### Train the model
Run the command `pipenv run python train.py`.

### Start the app
Run the command `pipenv run python chat.py`.

## What can I talk about?
Chatty is designed to carry out basic conversation, and is equipped with a collection of jokes and fun facts to share.
