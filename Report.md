# Learning to Debate

This report covers a project undertaken as part of COMP 421 - Machine Learning. The project involved the use of character wise recurrent neural networks to learn the structure of US presidential debates at a character level. This project makes use of [Karpathy's char-rnn project](https://github.com/karpathy/char-rnn) which provides implementation of three recurrent neural network systems implemented in Lua. The motivation for this project was the [DeepDrumpf](https://twitter.com/deepdrumpf) Twitter bot which generates tweets in the style of Donald Trump.

#### Collecting Data
The debate data for this project takes the form of text file transcripts collected from a couple of sources. The 2008 and 2012 presidential debates each consisted of four debates which were sourced, in text form, from the [Commission on Presidential Debates](http://www.debates.org/index.php?page=debate-transcripts). At the time of writing, the Commission has not made transcripts available for 2016's presidential debates. Conveniently, the Washington Post has created transcripts for the [first](https://www.washingtonpost.com/news/the-fix/wp/2016/09/26/the-first-trump-clinton-presidential-debate-transcript-annotated/), [second](https://www.washingtonpost.com/news/the-fix/wp/2016/10/09/everything-that-was-said-at-the-second-donald-trump-vs-hillary-clinton-debate-highlighted/), and [third](https://www.washingtonpost.com/news/the-fix/wp/2016/10/19/the-final-trump-clinton-debate-transcript-annotated/) debates of the 2016 election in a format which matches those from the Commission. When concatenated, the transcripts for these 11 debates amount to 1.2MB of data. The Shakespeare dataset provided by Karpathy for testing is 1.0MB so this debate dataset should be sufficient.

#### Infrastructure
On the default settings, a network took 2 to 3 hours to train on my gpu-less laptop. This wasn't going to work for all the different combinations of settings I wanted to try. AWS seemed a good solution to the problem of "Not enough compute power" so I created myself an account and followed a [set of instructions](https://github.com/brotchie/torch-ubuntu-gpu-ec2-install) provided by James Bortchie which outline how to get a server instance set up to run the char-rnn code. In fact, if anyone wants to re-run my analysis, you can simply follow the steps in that tutorial and swap out the final ```git clone``` stage with a clone of [my fork of the repository](https://github.com/LukeInkster/char-rnn/). I settled on a g2.2xlarge instance running in the Sydney datacenter which provides a Xeon CPU and a NVIDIA GPU with 1,536 cores and 4GB of VRAM. This brought the time to train a network on the default settings down to the 20 to 30 minute range, much more convenient for trying out a variety of combinations. I used a GitHub repository to manage moving code and results between the AWS instance and my development machine. This could have been achieved with through SSH or SFTP but I found it easier to keep everything grouped in a repository to prevent missing any files during transfers and git's compression and duplicate checking for file transfers was a nice bonus. 

## Which network debates best?
The code provided by Karpathy offers three different neural network models:
- Simple Recurrent Neural Network (RNN)
- Long Short Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

I want to get an idea of the differences between these networks and to find which performs best on the debate data. The differences between the networks gradually shrink as the training progresses, and after a full run of 50 epochs it's not possible to tell the difference between the quality of samples from each network. After fewer runs however, there are some noticeable differences between these samples. The following are some examples after two epochs of training for each network. 
#### RNN - Recurrent Neural Network
```
My..

I’d been who gon't deens.

I wastes.

ROMNEY: Ald us show, and Kertabelty, addieg one it a ladgues helaring. In John McCain of tickly, if. I salo incomely on is it. What saidicalitigation (plange the took atfory commonte is thinge, -- I sain in viliel inveriass beon. When fale araugn on that hond because?

WALLACE: think that the cover in Wermend that a seal are very-sige time. And let's grougj it going to the ward a poacrutiof a goest.

We've navory from it's what what, oo and He gates "owleds sere, nominit. And what the decanef the with al Qambownemal best to las. Lible. I'll neet eithifer, John McCain this dedent take such of investan. That's, I alto I have in think thank. They'll with certary liods holes offureving the tax now is to hele.

Now, agd orreaction. He I's the providing Ill) as didn't get and lot to gover and American with they one any tow thank, "an $2Okiss to adlencentwing ho’m. Senitaring invulling umericals.
```
#### LSTM - Long Short Term Memory
```
M’ aly be.

ERDTOT: That's other internowil this. And I what to have agrinablet. I lon't it it a mading, and so cries and the Amenial? And see us noblung to both the agring.

LLEINRE: Rund to ke the libted to has. Thuild president.

And I keable the just didinusination. Shered, abore way the Amediacy in Bedicate..

RLANATZ: Clat Bour in fumiil of ut hake cole that I'll, not's get to this incrovetioniant going to the wand Remmablry. And he wiples out text, let in whe what wind the Simia Alaq, having we'pland have thighersial seford.

MCCAIN: Let dible, with ne dither.

Bugher sajion the at a stration of is this who call has Adenawar is have no beon poitieg pror me. And "why thing then it's to deal right miraw great of., I’s the tare to distrorlicy sigut har prone the presicenct in this with mole want the deplated there it, said tax not if. us to weald ary tow to and -- I houst out a phormmetion, wy need asdo we've grong to then I'm. And what here, mifhion fath forith abso wover what is in the wored.
```
#### GRU - Gated Recurrent Network
```
M�Us

I’ll help. We need of the mids spost and toldible to making that whece'se that deal telred mades under the wlise offile Ofable, we've doard in to dollago. The people-way that.

IFMIDETROMK
. Chil, Senator Mmonez HeSsolom Romney secn teach.

Eremural Senator Bute progress plan a tax plen of NALWALE: I think you, and you san America. You just viol and the peeped.

We have have to cose fores worft our has now. I trahe Jire -- whee a hearth as prodraut of the issuate jobs are nich racal pers to dew-rately, counter toars for or.

And I sair that Americans no country of Amerecant of goeds beroud some or matsiels, in Iran, take such presention country wioh our Gelatas o-. We're caoring to heve for the peoples freend out duys, the tax groad to ore coopleun-back, we are going empereenment crosicel inveredible and the president of the offiiliative to you, herversy on Nort poicharsates, I think to progry, and I've renord your melt houdd us is up and this of the days is to getting the time lead brieds.

LEHRER: I fect to be nuy-defitire. And I looking who say, Dhabter, some peocge fuve of the high America lates, "ustion, I've got to good for the said-been intigarting. And Maracare, Jith, paidE. We're going to MeDady, betined healing this pusinie, buckoul tome in NACWasted in Iraq, Senigres trictions, and that is to reasuation. That's he bupp to he spendent plan -- it's a debute.
```
After two epochs, the networks have all learned some of the most basic rules about the format. They have each learned that line breaks always come in pairs, and that each line terminates in a full stop. They have also learned to start some lines with a name in capitals, then a colon, then a space, but are still not perfect at this and come up with names like "IFMIDETROMK". Overall, it is difficult to determine which network is providing the best output overall. The GRU network struggles with the overall structure but has a fair few correct words forming. The LSTM seems to have the best structure but relatively few of it's words are English (though many feel like they could be). The RNN seems somewhere in between on both metrics, approaching the correct structure and English words, but excelling in neither.

## Shakespearean Debates
I decided to see if I could train the network on both debates and on the Shakespeare data provided in with the network code. The first, and easiest to implement, approach I tried simply took a concatenation of the two files and ran this through the trainer as usual. I was expecting to get outputs which formed some Shakespearean-debate hybrid but the actual result was perhaps more interesting. I found the sample produced wildly different outputs for the same trained network depending on the seed. It seemed to slip into the format of either Shakespeare or a debate early on, then stick to this. Even the lanugage would match with whichever form it chose. Further experimentation found that the network could be primed with phrases with high density of terms expected of one of the formats but not in the other. The network would then continue on in that form for the rest of the sample. The following are two examples of this, sampling from the same network just with changes to the primetext parameter:

###### Command
```
$ th sample.lua lstmDebateShakespeareDefault/lm_lstm_epoch50.00_1.2423.t7 -primetext 'OBAMA: Making a claim about Iraq. Maybe somthing about war and the United States of America'
```

###### Output
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

###### Command
```
$ th sample.lua lstmDebateShakespeareDefault/lm_lstm_epoch50.00_1.2423.t7 -primetext $'ROMEO:\nTalking about thy king and thy castle'
```

###### Output
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

It seems the network has effectively learned two completely separate models which can be triggered with the right prime text. The network even maintains the Shakespearean format of line breaks after the speakers name and shorter lines in general when primed with talk of kings and castles. This is much different from the format of the sample when primed with talk of the United States of America and their wars.
