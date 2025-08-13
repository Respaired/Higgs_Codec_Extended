---
license: mit
tags:
- codec
- audio_tokenizer
- audio_codec
---

This is an on-going project. it is a modified version of Higgs-Boson audio tokenizer, you can fully train it. all scripts have been tested. 
a Few notes however:

1- this is not backward compatible with the original checkpoint (I think you can tweak it to be, but you have to adhere to Boson community license if you do.)
2- I highly recommend you to pretrain the model without the mel and adversarial setup first. it saves you a significant amount of compute and speed-up your convergence.
3- for the semantic teacher, I am using ```utter-project/mHuBERT-147``` which has a good multilingual support. if you want the original setup you can change it in the config.

I will train a checkpoint on a larger enough dataset one of these days after figuring out a few things first. but the setup is solid.

Happy training (~~inshallah~~).