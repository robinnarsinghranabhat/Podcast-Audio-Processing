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
Apparaenlty Model cannot find the pattern. Model is not learning. Currently using dialated convolution and went with kernel size of 7

For Rapid test, Idea : Loaded the Entier folder in google drive. Rapid Test Training there ! 

Other Ways : Train on the GRU based model Architecture on Coursera

Potential : Check Spectrogram masking idea in visual



