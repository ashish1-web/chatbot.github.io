# Building a ChatBot with Deep NLP
 
 
 
# Importing the libraries
import numpy as np                           #importing the numpy library
import tensorflow as tf                      #importing the tensorflow library   
import re                                    #library for replacing text in Natural Language processing and clean the text               
import time                                  # To measure the training time   
 
 
 
########## PART 1 - DATA PREPROCESSING ##########
 
 
 
# Importing the dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')    # to avoid the encoding issue # to ignore erros # to read dataset # to split the data in lines by the observaions
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')  
 
# Creating a dictionary that maps each line and its id
id2line = {}                                                 #For mapping each id with its Line
for line in lines:
    _line = line.split(' +++$+++ ')                          # We splitted with this string +++$+++
    if len(_line) == 5:                                      # In order to avoid the shifting issues and sacrifice some lines           
        id2line[_line[0]] = _line[4]                         # Mapping the id with the line  (L10006 YOU KNOW THAT I KNOW THAT)

# Creating a list of all of the conversations
conversations_ids = []                                         # We already have the conversation list but we create a new one by filtering all the meta data
for conversation in conversations[:-1]:                        #The exclude the last row which is empty
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")        #extracting only the converstion ids
    conversations_ids.append(_conversation.split(','))                                                 #Got the huge list of lists
 
# Getting separately the questions and the answers
questions = []                                                 #Creating the inputs and targets for inputting the training data
answers = []                                                   #Creating the inputs and targets for inputting the training data
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):                     #Iterating through the entire converstion lists
        questions.append(id2line[conversation[i]])             #The present key of the line is the question  
        answers.append(id2line[conversation[i+1]])             #The next key of the question is the answer
 
# Doing a first cleaning of the texts           
def clean_text(text):
    text = text.lower()                                       #Putting everything in the lower case
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text
 
# Cleaning the questions
clean_questions = []                                            # Storing the Questions in the clean_question List
for question in questions:
    clean_questions.append(clean_text(question))         
 
# Cleaning the answers
clean_answers = []                                             # Storing all the Answers in the clean_answers List
for answer in answers:
    clean_answers.append(clean_text(answer))
 
# Filtering out the questions and answers that are too short or too long
short_questions = []                                        # We are removing the non-frequent words so that we can optimize our training
short_answers = []
i = 0
for question in clean_questions:
    if 2 <= len(question.split()) <= 25:
        short_questions.append(question)
        short_answers.append(clean_answers[i])
    i += 1
clean_questions = []
clean_answers = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        clean_answers.append(answer)
        clean_questions.append(short_questions[i])
    i += 1
 
# Creating a dictionary that maps each word to its number of occurrences
# We are removing the non-frequent words so that we can optimize our training
word2count = {}
for question in clean_questions:
    for word in question.split():               #For getting the words directly
        if word not in word2count:
            word2count[word] = 1                #Count the number of times a word occurs in a question
        else:
            word2count[word] += 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:              #Count the number of times a word occurs in a answer
            word2count[word] = 1
        else:
            word2count[word] += 1
 
# Creating two dictionaries that map the questions words and the answers words to a unique integer
threshold_questions = 15                                   #We are doing tokenization and filtering of the data
questionswords2int = {}                                    #Mapping all the question words to a unique integer             
word_number = 0                                             
for word, count in word2count.items():                      #We can words and their counts within that same dictionary
    if count >= threshold_questions:                        # If the number of occurences is larger than the thrshold
        questionswords2int[word] = word_number              # Mapping that word into the dictionary where key is the word and the value is unique integer
        word_number += 1                                    # Incrementing the word id                        
threshold_answers = 15                                      #We are doing tokenization and filtering of the data
answerswords2int = {}                                       #Mapping all the answers words to a unique integer
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers:                          #We can words and their counts within that same dictionary
        answerswords2int[word] = word_number
        word_number += 1
 
# Adding the last tokens to these two dictionaries                  #<PAD> this token will be placed in place of all the spaces #<EOS> it is for denoting the end of string #<SOS> It is for denoting the start of string #<OUT> This token is used for denoting the occurene of words that are less than 5%
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']                           
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1        #Assingging the each token a unique integer
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1
 
# Creating the inverse dictionary of the answerswords2int dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}   #We will use this inverse mapping while buliding the seq2seq model
 
# Adding the End Of String token to the end of every answer
for i in range(len(clean_answers)):                                #Attaching the <EOS> Token at the end of the string
    clean_answers[i] += ' <EOS>'
 
# Translating all the questions and the answers into integers
# and Replacing all the words that were filtered out by <OUT> 
questions_into_int = []
for question in clean_questions:
    ints = []                                               
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)
 
# Sorting questions and answers by the length of questions
sorted_clean_questions = []                                                 # done because to speed up the training and to minimize data loss # Reduce the amount of padding in the training
sorted_clean_answers = []
for length in range(1, 25 + 1):                                            # Shortest question 1 and the longest question 25                     
    for i in enumerate(questions_into_int):                                # To make a couple of index and the question quesion itself
        if len(i[1]) == length:                                            # Sorting the quesions and the answers accoriding to the length of the questions
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
 
 


 
########## PART 2 - BUILDING THE SEQ2SEQ MODEL ##########
 
 
 
# Creating placeholders for the inputs and the targets                               #All the variables used in the tensors must be defined as tensor variables known as instances 
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')                 # Will be containg all the inputs # Questions encoding into integers # Inputs are two dimentional matrix
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')               # Will be containg all the Targets # Questions encoding into integers # Targets are two dimentional matrix
    lr = tf.placeholder(tf.float32, name = 'learning_rate')                         # Will hold the learning rate which is a hyper-Parameter 
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')                      # HyperParameter which will control the dropout rate # RAte of the neurons which you want to overwite with the each iteration
    return inputs, targets, lr, keep_prob                                           # Returning all the parameters
 
# Preprocessing the targets                                                         #  The targets must be into batches for the decoder to accept them
def preprocess_targets(targets, word2int, batch_size):                              # Will return the targets , word2int to get the identifiew of the SOS TOKEN , take the batch size
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])                         # To create a matrix of the number of rows of the batch size and the number of columns as 1 and the second argument is the special integer to the sos token
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])          #the strided_slice basically extracts the subset of the tenor # first argument is the tesor # next argument is from where we we want to start extraction # third argument is till where we have to make the extraction #to mention the slide of one
    preprocessed_targets = tf.concat([left_side, right_side], 1)                    # To concat the tensors # first argument is the couple which we want to concat # for horizontal concatination 1 # for vertical contatination we take 0
    return preprocessed_targets
 
# Creating the Encoder RNN                                                                          #Creating the encoding layer of the 
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):                      # rnn_inputs is the models inputs such as the learning inputs , keep_prob # rnn_size is the number of input tensors of the encoder rnn layer # num_layers are the number of layers of the rnn #keep prob that controlls the dropout rate # list  of the lengths of each question in the batch
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)                                                   # Creating the Lstm #the number of input tensors in the layers (rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)                 # lstm_dropout to apply the dropout which we are getting from the arguments
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)                         # to specify the number of layers we want in our rnn encoder # To create the encoder cell # The encoder cell consists of many lstm layers
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,         # We have got our encoding cell now we will take care of our encoding state   
                                                                    cell_bw = encoder_cell,         # The input size of the forward cell and the backward cell must match
                                                                    sequence_length = sequence_length,  
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state
 
# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    #First argument will be the encoder state , decoder_cell in the rrnn of the decoder # inputs on which we are applying embedding # embedding is the
    # mapping from the words to vectors of real numbers each one encoding uniquely the word associated to it # taking the sequence length 
    #decoding_scope is a advanced datastructure that will wrap the  tensor flow variables
    # output_function is the function that we will use to  return the output of the  decoder_training set 

    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])

    # we will initilazie them as three dimentional matrices of 0s # no of lines will be the batch size  # no of columns will be one , the number elements on the third axis will be decode cell output

    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)

    #preprocessing the attention inputs # we have taken a particular attention option
    #attention keys are the keys that are to be compared with the target states
    #attention_values are the values that we will use to construct the context vectors.
    #attention_score function is to compare the similarities between the keys and the target states
    #attention_construct funtion is the function used to construct the attention states

    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train") # name scope of the decoder function
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
 
# Decoding the test/validation set(this function will be used to predict the validation set)
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
 
# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:                       #
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)        #creating the weights with a standard deviation of 0.1
        biases = tf.zeros_initializer()                                # creating the biases of the fully connected layer
        output_function = lambda x: tf.contrib.layers.fully_connected(x,            # output function 
                                                                      num_words,    #num of outputs 
                                                                      None,        # normalizer none
                                                                      scope = decoding_scope, #assign the weights
                                                                      weights_initializer = weights,  #assign the bia
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
 
# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    #inputs questions of the dataset after training these will be the questions that we will ask to the chatbot
    #targets are the answers that we have 

    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
 
 
 
########## PART 3 - TRAINING THE SEQ2SEQ MODEL ##########
 
 
 
# Setting the Hyperparameters
epochs = 100           #epochs is the whole process of getting the batches of inputs in the neural network and forward propogating themm inside the encoders and the encoder states and forward propogating the encoder states with the tragets inside the decoder of the neural network to get the output and backpropating it back to the neural to calculate loss and regenerate weights
batch_size = 32        # the batch size more the time consumed is less for training the neural network 
rnn_size = 1024        # size of the rnn = 1024
num_layers = 3         # number of layers = 3
encoding_embedding_size = 1024   # the number of columns in the encoding_embedded matrix that is the number of column we wamt to have for the embedding values where each line corresponds to each question
decoding_embedding_size = 1024   
learning_rate = 0.001           # It must not be too high not to be to low
learning_rate_decay = 0.9       # To learn in more depth the human conversations
min_learning_rate = 0.0001      
keep_probability = 0.5          # We have deactivated half of the neuron to prevent it from the overfitting # This reading has been taken from the paper of the jeffrey hinton
 
# Defining a session             
tf.reset_default_graph()                      #Here we are resetting the tensorflow session #Resetting the tensorflow graph
session = tf.InteractiveSession()             # Starting the new session which we will use for the training 
 
# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()      #we will load the input variables
 
# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')  # we are creating the sequence lengths  # We wount be using questions and answers which are having a length of more than 25
 
# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)                  #we need to get the shape of the tensor
 
# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),  # To reshape the tensor and reverse its dimensions
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)
 
# Setting up the Loss Error, the Optimizer and Gradient Clipping(this is a operation that will cap the gradients in the graph between the minimum value and the maximum value and that to avoid some vanishing gradients issues)
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length])) #to initilaze the shape of the tesor of weights
    optimizer = tf.train.AdamOptimizer(learning_rate)       # making an object of the optmizer
    gradients = optimizer.compute_gradients(loss_error)     # compute the gradiets with respect to the weights of the each neuron
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients) #optimized clipped gradients
 
# Padding the sequences with the <PAD> token
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
 
# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
 
# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]
 
# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")
 
 
 
########## PART 4 - TESTING THE SEQ2SEQ MODEL ##########
 
 
 
# Loading the weights and Running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)
 
# Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
 
# Setting up the chat
while(True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersints2word[i] == 'i':
            token = ' I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)
