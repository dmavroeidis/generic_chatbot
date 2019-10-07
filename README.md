# A toy chatbot
A toy chatbot using Attention, Dropout, Teacher Forcing, Pretrained Embeddings

### Short description 
This codebase uses the same model as the tutorial only replacing the existing embedding layer with pre-trained embeddings from google (GloVe). This change improved the performance of the chatbot as it catches word similarities between the input and the output. The embedding layer both in the encoder and the decoder, making all necessary changes in the model for the mechanism to work.

### About the code
The original code was split into separate modules: data, decoder, model. 
The data module contains the Voc class which is used to store information about the corpus such as what word corresponds to what index, as well as methods for adding data to it and trimming excess words. 
The model module contains all classes and methods relevant to the structure of the RNN model. 
The decoder module contains one class of the greedy decoder’s implementation.



### Acknowledments
The code in this project was adapted from a [PyTorch tutorial for building a chatbot](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html) by Matthew Inkawhich. A method for [using pretrained GloVe embeddings](https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76) by Martin Pellarolo was also used. The dataset used is the [MetalWOz](https://www.microsoft.com/en-us/research/project/metalwoz/) by Microsoft.
