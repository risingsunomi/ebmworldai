# EBM World AI
Experiments in using ai agent world model development using energy based models. 
Learning as I go but will add in some interesting findings here.

## Background
I am not a PHd or any student (dropped out of CS undergrad) but been doing software development since 2005, reading research papers and [did some early ML CV stuff](https://pitchbook.com/profiles/company/343232-83) 

After listening to the use of EBMs with world base models from [this podcast](https://www.youtube.com/watch?v=5t1vTLU7s40) I have decided to try it.

Mostly pulling from the [foundation paper](https://worldmodels.github.io/) I am switching out the MDN step and adding in a EBM.

So would like

```
 # old model 
 Environment -> ( VAE ) -> (EBM-LTSM) -> Action -> Environment

 # current model 03202024
 Environment -> AlexNet -> Energy via NCE loss -> ?
```

Lets see if a regular dude like me can pull it off.

## Setup
More details coming soon but for now user requirements.txt

Need to have tesseract-ocr

## Discussion
Come chat with me and other developers exploring the AI agent space
https://discord.gg/UWd6u5aR
