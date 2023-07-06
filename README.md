# sign-language-recognition

sign-language-recognition is a web app that uses Hidden Markov Models to train signed words. It uses a multidimensional Viterbi Trellis to predict a word using hand landmark data from MediaPipe. Words can be trained through the REST API implemented to access the Viterbi Trellis and the algorithm that divides training vectors into states. The API is used in the static webpages in ```django/templates``` using AJAX.

#### Words
The ```Words``` class stores the training vectors of all words trained, and every time ```Words.update_word()``` is called, states, prior_probs, emission_paras, and transition_probs are updated to include the new word or add training vectors to a trained word. ```Words.num_dimensions``` stores the number of data points and dimensions the Words object should store. It is also used to check a vector using the Viterbi Trellis. The Pickle library is used to store a Words object in the main file as well as in the Django project as a ```.pkl``` file.
#### MediaPipe
The ```HandTracker``` class stores data about the hand location in a specific frame. It uses MediaPipe's position finder and stores the x and y coordinates of 28 hand landmarks.
#### Viterbi Trellis
```multidimensional_viterbi()``` in ```convert.py```, which is used by ```Words.check_words()```, takes in the evidence_vector, states, prior_probs, transition_probs, and emission_paras, and returns the most likely word associated with the evidence vector as well as the probability of the association. It uses Hidden Markov Models and computes a Gaussian Probability to determine which state the evidence vector begins in.
#### Division into states
```average()```, ```sd1()```, ```convert()```, and ```divide()``` in ```convert.py``` are used to take input training vectors for a word and return a list of vectors divided into three states along with the emission parameters of each state. While it currently returns only three states, it can be modified to divide the training vectors into a changeable number of states, depending on the gesture recorded.
#### train.py
```train_word()``` and ```test_words()``` in ```train.py``` use OpenCV for real-time video input and HandTracker for detecting hand landmark positions. ```train_words()``` builds training vectors based on num_iterations and train_time, and calls ```Words.update_word()``` to store the new word or add vectors to an existing word. ```test_words()``` checks every input vector it builds based on train_time and calls ```Words.check_word()``` to predict the most likely word signed.
#### Django
The Django project contains an API (using Django REST Framework) to train and test words similar to the functions in ```train.py```, detecting hand landmarks using JavaScript's MediaPipe library instead. ```train_word()```, a POST method, takes a single vector along with the word name to train an existing word or add a new one. The JavaScript code in ```new_train.html``` calls it every time a new vector is formed using the detected hand landmarks. ```check_word()```, a GET method, takes an input vector, calls ```Words.check_word()```, and returns the predicted word. The JavaScript code in new_test.html calls it after forming an input vector depending on the number of hand landmarks detected.

### Running the Application
Install the requirements:<br>
````
pip install -r requirements.txt
````
Run the development web server:
````
python manage.py runserver
````
Open the URL ```http://localhost:8000/train-cam``` to train and ```http://localhost:8000/train-cam``` to predict.

