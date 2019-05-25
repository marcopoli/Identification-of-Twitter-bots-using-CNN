# Identification of bot accounts in Twitter using 2D CNNs on user-generated contents

*Marco Polignano, Marco Giuseppe de Pinto* 

*marco.polignano@uniba.it, marcogiuseppe.depinto@uniba.it*

# Abstract

The number of accounts that autonomously publish contents on the web is growing fast, and it is very common to encounter them, especially on social networks. They are mostly used to post ads, false information, and scams that a user might run into. Such an account is called bot, an abbreviation of robot (a.k.a. social bots, or sybil accounts). In order to support the end user in deciding where a social network post comes from, bot or a real user, it is essential to automatically identify these accounts accurately and notify the end user in time. In this work, we present a model of classification of social network accounts in humans or bots starting from a set of one hundred textual contents that the account has published, in particular on Twitter platform. When an account of a real user has been iden- tified, we performed an additional step of classification to carry out its gender. The model was realized through a combination of convolutional and dense neural networks on textual data represented by word embedding vectors. Our architec- ture was trained and evaluated on the data made available by the PAN Bots and Gender Profiling challenge at CLEF 2019, which provided annotated data in both English and Spanish. Considered as the evaluation metric the accuracy of the sys- tem, we obtained a score of 0.9182 for the classification Bot vs. Humans, 0.7973 for Male vs. Female on the English language. Concerning the Spanish language, similar results were obtained. A score of 0.9156 for the classification Bot vs. Hu- mans, 0.7417 for Male vs. Female, has been earned. We consider these results encouraging, and this allows us to propose our model as a good starting point for future researches about the topic when no other descriptive details about the ac- count are available. In order to support future development and the replicability of results, the source code of the proposed model is available on the following GitHub repository: https://github.com/marcopoli/HumanOrBot_try_to_guess

# Instructions

- `runner.py` has been used to run and submit the model to the competition
- `female_es.txt` and `male_es.txt` contain additional tweets that have been scraped using `datagen.py` and `datagen_es.py` to create additional training data.
- The `genHuman` folder contains the tweets created with the `datagen` files.
- `malefemale.py` and `malefemale_es.py` create/train the model to distinguish between male and female tweets (if human and not bot).
- `botnotbot.py` and `botnotbot_es.py` create/train the model to distinguish between human and bot tweets.

# Dataset

The dataset is under copyright of the PAN 2019 challenge. 
It can be used only for research purposes citing the owner: https://pan.webis.de/clef19/pan19-web/author-profiling.html

Download link: https://my.pcloud.com/publink/show?code=XZ2kIK7ZWtVHRwpxKVb0VYjLM6MewkXWrMn7