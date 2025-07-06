# -*- coding: utf-8 -*-
"""LSTM_NextWord_Prediction.ipynb
"""

data = """ About the Program
What is the fee for the Full Stack Web Development Bootcamp?
The bootcamp follows a monthly subscription model where you pay Rs 999/month.
What is the total duration of the bootcamp?
The total duration of the bootcamp is 6 months, so the total fee becomes 999*6 = Rs 6000 (approx).
What is the syllabus of the bootcamp?
We cover the following modules:
HTML & CSS
JavaScript (Basic to Advanced)
Version Control with Git & GitHub
React.js
Node.js & Express
MongoDB & SQL
REST APIs
Authentication & Authorization
Deployment using Vercel & AWS
You can check the detailed syllabus here - https://example.com/fullstack-syllabus
Will DevOps or CI/CD be covered in this bootcamp?
Only introductory topics will be covered; the main focus is on MERN stack development.
What if I miss a live session? Will I get a recording?
Yes, all sessions are recorded and uploaded to your dashboard.
Where can I find the class schedule?
Check the official schedule sheet here - https://example.com/schedule
What is the average duration of a live session?
Approximately 2.5 hours per session.
What is the mode of instruction?
Hinglish (mix of Hindi and English)
How will I be informed about the upcoming class?
You will get an email and WhatsApp reminder before each session.
Can I join the course if I have no technical background?
Yes, the bootcamp is beginner-friendly.
Is it possible to join in the middle of the bootcamp?
Absolutely. Once you join, all past sessions will be unlocked.
Do I need to submit assignments?
No need to submit. Solutions will be provided for self-evaluation.
Will we build real projects in the course?
Yes, each module includes a capstone project.
Where can I contact for general support?
You can write to support@webcamp.in
Payment/Registration Questions
Where should I make my payments?
All payments must be done on our official website - https://example.com
Can I pay the full fee in one go?
No, the bootcamp follows a monthly subscription model only.
What is the validity of each monthly payment?
30 days from the date of payment.
Is there a refund policy?
Yes, we offer a 5-day refund policy from the day of payment.
I live outside India and can’t make a payment, what should I do?
Please write to us at support@webcamp.in for international payment options.
Post Registration Queries
How long will I have access to paid content?
Till your subscription is active. After completing all 6 payments, access is extended till Dec 2025.
Why is lifetime access not provided?
Due to the low course fee, we can’t offer lifetime access.
How can I clear doubts?
Submit the doubt form available in your dashboard to schedule a 1-on-1 session.
Can I ask doubts from earlier sessions if I join late?
Yes, just select the correct session/week in the doubt form.
Certificate & Placement Assistance
What is the criteria to get the certificate?
1. Complete payment (Rs 6000 total)
2. Attempt at least 80% of the assignments
What does placement assistance include?
Portfolio building
Mock interviews
Resume review
Referral network access
Job hunting strategy sessions
Does it guarantee a job or interview?
No, we do not guarantee placements or interviews. Assistance is provided, not assurance."""

pip install Tokenizer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tokanizer = Tokenizer()

tokanizer.fit_on_texts([data])

tokanizer.word_index

for sentence in data.split('\n'):
  print(sentence)

input_seq = []

for sentence in data.split('\n'):
  tokanized_sentence = tokanizer.texts_to_sequences([sentence])[0]

  for i in range(1, len(tokanized_sentence)):
    n_gram_seq = tokanized_sentence[:i+1]
    input_seq.append(n_gram_seq)

input_seq

max_length = max([len(x) for x in input_seq])

max_length

from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_input_seq = pad_sequences(input_seq, maxlen=max_length, padding='pre')
display(padded_input_seq)

X = padded_input_seq[:, :-1]
y = padded_input_seq[:, -1]

print(X)

print(y)

X.shape

y.shape

len(tokanizer.word_index)

from tensorflow.keras.utils import to_categorical

y = to_categorical(y, num_classes=len(tokanizer.word_index)+1)

y.shape

y

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()

model.add(Embedding(251, 100, input_length=19))
model.add(LSTM(150))
model.add(Dense(251, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()

model.fit(X,y,epochs=100)

import time
text = "We cover"

for i in range(10):
  # tokenize
  token_text = tokanizer.texts_to_sequences([text])[0]
  # padding
  padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
  # predict
  pos = np.argmax(model.predict(padded_token_text))

  for word,index in tokanizer.word_index.items():
    if index == pos:
      text = text + " " + word
      print(text)
      time.sleep(2)

