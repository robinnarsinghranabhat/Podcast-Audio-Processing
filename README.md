# Podcast-Audio-Processing
Experiment on ways to make listening podcast better

Raw learnings maintained at : 
https://docs.google.com/document/d/1gaa6yNEmFcRK9GH3IdASxcCh8qLXrKyfxRyarOoeW90/edit?usp=sharing

## Problem in general 

Control the podcast played in my phone, laptops through my voice input. Stop, Rewind, Take note at that point seperately and continue.



### Potential Learning curve to building this 



1)  General Project Structure planning 

    https://www.samueldowling.com/2020/06/08/how-to-set-up-a-python-project-and-development-environment/

    - Working with a makefile    
    https://stackabuse.com/how-to-write-a-makefile-automating-python-setup-compilation-and-testing/
    - Test, linters, style formatter, documentation, version    control ...

2) Technical Aspect

    - What does it mean to understand sound :+1:
    - Basics of Audio processing  :+1: 
    - Could the problem be simpler in my case .. Could simple digital electornics get me by ?
    - Deep learning to understand patterns 


Completed Till now : 
- Data preparation (TODO : Automate audacity for trimming)
- Threaded Inference
- Adding Audio Dataloaders
- Dummy Run, check loss
- Data Augmentaion with Torch Audio


Latest Update : 
- [+1] Merge audio samples with variable Gains 
- [] Augmentation : Modify Gain of Traning set during augmentation  (pydub part) 
    - Try using Pytorch for this.
- [] For Rapid test, Idea : Loaded the Entier folder in google drive. Rapid Test Training there ! 
- [] Apparaenlty Model cannot find the pattern. Model is not learning. Currently using dialated convolution and went with kernel size of 7
- [] Other Ways : Train on the GRU based model Architecture on Coursera <-- Just use this architecture

- Potential : Check Spectrogram masking idea in visual



Traning Updates and todo : 
Updates
- With larger kernel size, than used in normal 28*28 image, and some BatchNorm, Model is finally learning .
- But, is my model really catching that pattern ??  like, is it really responding to activate and not anything else in that whole spectrogram.  
  For this, Now force model to output multiple sigmoiuds
- What if, Model is learning, at sudden spike from my word, and not activate in general. So I need to put negative snippet of my vocies at negative examples to blur out this possiblity

Todo
- Train in Colab, Larger model, larger dataset, Add Negative keywords and my own random noise in negative examples
- Larger model, More augmentation FreqMask, 
- Internal Torch for Augmentation, and more workers in augmentation 
- 
- Add other keywords like, Pause, Note to get Detected. 
- Make Traning more difficult. Like, p?0.8 and not 0.5 to account for positive detection in training phase.


Inference Update : Make inference more real timish

- MultiThreading is not really working. Basically, it's not contuonusly saving recordings in background in real time. Have to try MULTIPROCESSING to just deposit the inputs / Saving the inputs. And while our Pytorch loads it, handle exception / sleep timer if it's loading incompletely saved audio. So, like, 

    - Make two processes, one records and saves , while, other infers
    - Put them in sync such that, p1 = pool(Record) , p2 = pool(INfer) , p1.join() p2.join() .. don't know

- 



