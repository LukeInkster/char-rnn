# Learning to Debate

This report covers a project undertaken as part of COMP 421 - Machine Learning. The project involved the use of character wise recurrent neural networks. This project makes use of 

#### Collecting Data
The debate data for this project takes the form of text file transcripts collected from a couple of sources. The 2008 and 2012 presidential debates each consisted of four debates which were sourced, in text form, from the [Commission on Presidential Debates](http://www.debates.org/index.php?page=debate-transcripts). At the time of writing, the Commission has not made transcripts available for 2016's presidential debates. Conveniently, the Washington Post has created transcripts for the [first](https://www.washingtonpost.com/news/the-fix/wp/2016/09/26/the-first-trump-clinton-presidential-debate-transcript-annotated/), [second](https://www.washingtonpost.com/news/the-fix/wp/2016/10/09/everything-that-was-said-at-the-second-donald-trump-vs-hillary-clinton-debate-highlighted/), and [third](https://www.washingtonpost.com/news/the-fix/wp/2016/10/19/the-final-trump-clinton-debate-transcript-annotated/) debates of the 2016 election in a format which matches those from the Commission. When concatenated, the transcripts for these 11 debates amount to 1.2MB of data. The Shakespeare dataset provided by Karpathy for testing is 1.0MB so this debate dataset should be sufficient.

#### Testing in a sensible amount of time


## Which Recurrent Neural Network debates best?
#### RNN
#### LSTM
#### GRU



## Shakespearean Debates


```
$ th sample.lua lstmDebateShakespeareDefault/lm_lstm_epoch50.00_1.2423.t7 -primetext 'OBAMA: Making a claim about Iraq. Maybe somthing about war and the United States of America'
```

```
OBAMA: Making a claim about Iraq. Maybe somthing about war and the United States of America. He said my jobs because I have gone to create might be wisely not getting their city one fortholate, under $250,000 a year. The Lating Reagat live, she didn't have closing Americans for them and intellect small businesses.

TRUMP: So -- look, and I’ve the prices. You have to be ruled and part of the plumber. We've got to tell the same naturals.

And of you know that they looked at 2014 press will have just regulations that we don't have and off-shore and in the God...

MCCAIN: No, it, if it gets Prilations.

CLINTON: Well, let me get $90 billion of times low to work.

We should you ask us alone.

But in the watched we wencot his father Plan is, it will help loon in a very founder legislative the last country.

OBAMA: OK.

OBAMA: About that our centurous, because I can heal higher seems about nuclear sould making chemelusly.

I would putter the campaign of wantor regardles aren.

LEHRER: I think God as a governor of Republicans rempsions on a balanced budget so reason ab, yes, contract us to grow to...
```

```
$ th sample.lua lstmDebateShakespeareDefault/lm_lstm_epoch50.00_1.2423.t7 -primetext $'ROMEO:\nTalking about thy king and thy castle'
```

```
ROMEO:
Talking about thy king and thy castle;
But my kinsman's inform and so in our master.

HERMIONE:
Thou gave before you, man; let's govern'd my life,
Give you more resord his day's graps. And more
queer, mine, sir, be loves his mank'd,
Biddleam castle thempens! Pray, and not County lie.
Thus shall not better'd if you call what I know
The dust shut a piteous important,
Lest thou ignoring a color knee she:
It was lost for his and any migrak.

Lord:
We'll yet, let such serving kinsman:
What is my chair as we have members. Less
Proed this entrailant, problems!
Culling you, for deaths!
```
