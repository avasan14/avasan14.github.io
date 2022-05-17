---
layout: post
title: Implementing Tensorflow to Develop a Fake News Classifier
---

Rampant misinformation — often called “fake news” — is one of the defining features of contemporary democratic life. I will develop and assess a fake news classifier using Tensorflow.

## Imports

Here are the following packages that will be used in order to develop the fake news classifier:

```python
import numpy as np
import pandas as pd
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow import keras
import re
import string
from matplotlib import pyplot as plt
from tensorflow.keras import layers, losses, utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import plotly.express as px 
from sklearn.decomposition import PCA
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


## Acquiring the Data

The following code is used to acquire the data:

```python
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
fake_news = pd.read_csv(train_url)
```

## Making a Dataset

I will write a function called `make_dataset`. This function should do two things:

1. Remove stopwords from the article text and title. A stopword is a word that is usually considered to be uninformative, such as “the,” “and,” or “but.”
2. Construct and return a tf.data.Dataset with two inputs and one output. The input should be of the form `(title, text)`, and the output should consist only of the `fake` column.

```python
stop_words = stopwords.words('english')

def remove_stopwords(texts):
  return [' '.join([word for word in str(doc) if word not in stop_words]) for doc in texts]

def make_dataset(df):
  data['title'] = remove_stopwords(data['title']) #remove stopwords from titles
  data['text'] = remove_stopwords(data['text']) #remove stopwords from text
  # create Dataset using title and text without stopwords
  data = tf.data.Dataset.from_tensor_slices( #process it into a tensorflow data
      (
        {
            "title" : data[["title"]], 
            "text" : data[["text"]]
        }, 
        {
            "fake" : data["fake"]
        }
    )
  )
  # set batch to 100 for faster runtime 
  my_data_set.batch(100)
  return my_data_set
```

```python
fake_news = make_dataset(fake_news)
```

### Creating Validation data

After constructing my primary `Dataset`, I will split off 20% of it to use for validation.

```python
# random shuffle of the data
fake_news = fake_news.shuffle(buffer_size = len(fake_news))
# 70% train, 20% validation, 10% test
train_size = int(0.7*len(fake_news)) 
val_size = int(0.2*len(fake_news))

train = fake_news.take(train_size)
val = fake_news.skip(train_size).take(val_size) 
test = fake_news.skip(train_size + val_size)
```

### Finding the Base Rate

```python
iterator= train.unbatch().map(lambda x, fake: fake).as_numpy_iterator()
count = 0

for i in range(len(train)):
  # adds 1 if article is fake and 0 if article is real
  count += iterator.next()["fake"]
print(count)
```
    8265

```python
base_rate = count/len(train)
base_rate
```
    0.5259641084383352

We can see that the base rate of the training dataset is approximately 0.526. This means that if the model were to solely guess fake news, then it would be correct ~52.6% of the time.

### Text Vectorization

We will create a text vectorization that will be used when creating the models in the following parts. This layer should be adapted to both the article title and article text.

```python
#preparing a text vectorization layer for tf model
size_vocabulary = 2000

def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation 

vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 

vectorize_layer.adapt(train.map(lambda x, y: x["title"]))
vectorize_layer.adapt(train.map(lambda x, y: x["text"]))
```

## Creating Models

In my first model, I will use **only the article title** as an input.

```python
title_input = keras.Input(
    shape=(1,),
    name = "title", # same name as the dictionary key in the dataset
    dtype = "string"
)
```

```python
title_features = vectorize_layer(title_input) # apply this "function TextVectorization layer" to title_input
title_features = layers.Embedding(size_vocabulary, output_dim = 3, name="embedding")(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(32, activation='relu')(title_features)

output = layers.Dense(2, name = "fake")(title_features)
```

```python
model1 = keras.Model(
    inputs = title_input,
    outputs = output
)

model1.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)

history = model1.fit(train, 
                     validation_data=val,
                     epochs = 25)
```

    Epoch 1/25


    /usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:559: UserWarning: Input dict contained keys ['text'] which did not match any model input. They will be ignored by the model.
      inputs = self._flatten_to_reference_inputs(inputs)


    15714/15714 [==============================] - 103s 6ms/step - loss: 0.3699 - accuracy: 0.8110 - val_loss: 0.1385 - val_accuracy: 0.9472
    Epoch 2/25
    15714/15714 [==============================] - 98s 6ms/step - loss: 0.1277 - accuracy: 0.9511 - val_loss: 0.0898 - val_accuracy: 0.9717
    Epoch 3/25
    15714/15714 [==============================] - 99s 6ms/step - loss: 0.1119 - accuracy: 0.9596 - val_loss: 0.0606 - val_accuracy: 0.9811
    Epoch 4/25
    15714/15714 [==============================] - 96s 6ms/step - loss: 0.1020 - accuracy: 0.9627 - val_loss: 0.0567 - val_accuracy: 0.9804
    Epoch 5/25
    15714/15714 [==============================] - 96s 6ms/step - loss: 0.0946 - accuracy: 0.9640 - val_loss: 0.0542 - val_accuracy: 0.9788
    Epoch 6/25
    15714/15714 [==============================] - 96s 6ms/step - loss: 0.0951 - accuracy: 0.9649 - val_loss: 0.0806 - val_accuracy: 0.9717
    Epoch 7/25
    15714/15714 [==============================] - 92s 6ms/step - loss: 0.0902 - accuracy: 0.9665 - val_loss: 0.0398 - val_accuracy: 0.9866
    Epoch 8/25
    15714/15714 [==============================] - 93s 6ms/step - loss: 0.0881 - accuracy: 0.9681 - val_loss: 0.0457 - val_accuracy: 0.9844
    Epoch 9/25
    15714/15714 [==============================] - 94s 6ms/step - loss: 0.0836 - accuracy: 0.9671 - val_loss: 0.0454 - val_accuracy: 0.9813
    Epoch 10/25
    15714/15714 [==============================] - 93s 6ms/step - loss: 0.0801 - accuracy: 0.9703 - val_loss: 0.0463 - val_accuracy: 0.9833
    Epoch 11/25
    15714/15714 [==============================] - 96s 6ms/step - loss: 0.0817 - accuracy: 0.9686 - val_loss: 0.0436 - val_accuracy: 0.9853
    Epoch 12/25
    15714/15714 [==============================] - 93s 6ms/step - loss: 0.0796 - accuracy: 0.9695 - val_loss: 0.0488 - val_accuracy: 0.9806
    Epoch 13/25
    15714/15714 [==============================] - 92s 6ms/step - loss: 0.0765 - accuracy: 0.9700 - val_loss: 0.0474 - val_accuracy: 0.9804
    Epoch 14/25
    15714/15714 [==============================] - 93s 6ms/step - loss: 0.0761 - accuracy: 0.9715 - val_loss: 0.2642 - val_accuracy: 0.8940
    Epoch 15/25
    15714/15714 [==============================] - 95s 6ms/step - loss: 0.0747 - accuracy: 0.9707 - val_loss: 0.0434 - val_accuracy: 0.9849
    Epoch 16/25
    15714/15714 [==============================] - 100s 6ms/step - loss: 0.0794 - accuracy: 0.9700 - val_loss: 0.0613 - val_accuracy: 0.9771
    Epoch 17/25
    15714/15714 [==============================] - 99s 6ms/step - loss: 0.0744 - accuracy: 0.9709 - val_loss: 0.0530 - val_accuracy: 0.9791
    Epoch 18/25
    15714/15714 [==============================] - 99s 6ms/step - loss: 0.0771 - accuracy: 0.9713 - val_loss: 0.0396 - val_accuracy: 0.9844
    Epoch 19/25
    15714/15714 [==============================] - 109s 7ms/step - loss: 0.0673 - accuracy: 0.9747 - val_loss: 0.0426 - val_accuracy: 0.9849
    Epoch 20/25
    15714/15714 [==============================] - 99s 6ms/step - loss: 0.0736 - accuracy: 0.9710 - val_loss: 0.0412 - val_accuracy: 0.9864
    Epoch 21/25
    15714/15714 [==============================] - 100s 6ms/step - loss: 0.0743 - accuracy: 0.9721 - val_loss: 0.0903 - val_accuracy: 0.9617
    Epoch 22/25
    15714/15714 [==============================] - 99s 6ms/step - loss: 0.0735 - accuracy: 0.9718 - val_loss: 0.0765 - val_accuracy: 0.9704
    Epoch 23/25
    15714/15714 [==============================] - 100s 6ms/step - loss: 0.0686 - accuracy: 0.9724 - val_loss: 0.0401 - val_accuracy: 0.9851
    Epoch 24/25
    15714/15714 [==============================] - 101s 6ms/step - loss: 0.0715 - accuracy: 0.9722 - val_loss: 0.0375 - val_accuracy: 0.9857
    Epoch 25/25
    15714/15714 [==============================] - 102s 7ms/step - loss: 0.0677 - accuracy: 0.9731 - val_loss: 0.0456 - val_accuracy: 0.9826

```python
# plot training and validation accuracy across epochs
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```
![fake_news_hw_24_1.png](/images/fake_news_hw_24_1.png)
    
In my second model, I will use **only the article text** as an input.

```python
text_input = keras.Input(
    shape=(1,),
    name = "text", # same name as the dictionary key in the dataset
    dtype = "string"
)
```

```python
text_features = vectorize_layer(text_input) # apply this "function TextVectorization layer" to text_input
text_features = layers.Embedding(size_vocabulary, output_dim = 3, name="embedding")(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(32, activation='relu')(text_features)

output = layers.Dense(2, name = "fake")(text_features)
```

```python
model2 = keras.Model(
    inputs = text_input,
    outputs = output
)

model2.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)

history = model2.fit(train, 
                     validation_data=val,
                     epochs = 25)
```

    Epoch 1/25


    /usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:559: UserWarning: Input dict contained keys ['title'] which did not match any model input. They will be ignored by the model.
      inputs = self._flatten_to_reference_inputs(inputs)


    15714/15714 [==============================] - 68s 4ms/step - loss: 0.2107 - accuracy: 0.9164 - val_loss: 0.0957 - val_accuracy: 0.9775
    Epoch 2/25
    15714/15714 [==============================] - 64s 4ms/step - loss: 0.1047 - accuracy: 0.9646 - val_loss: 0.0685 - val_accuracy: 0.9826
    Epoch 3/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0881 - accuracy: 0.9709 - val_loss: 0.0539 - val_accuracy: 0.9828
    Epoch 4/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0796 - accuracy: 0.9715 - val_loss: 0.0460 - val_accuracy: 0.9855
    Epoch 5/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0724 - accuracy: 0.9740 - val_loss: 0.0455 - val_accuracy: 0.9884
    Epoch 6/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0694 - accuracy: 0.9765 - val_loss: 0.0420 - val_accuracy: 0.9911
    Epoch 7/25
    15714/15714 [==============================] - 63s 4ms/step - loss: 0.0663 - accuracy: 0.9767 - val_loss: 0.0405 - val_accuracy: 0.9909
    Epoch 8/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0613 - accuracy: 0.9772 - val_loss: 0.0359 - val_accuracy: 0.9935
    Epoch 9/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0619 - accuracy: 0.9795 - val_loss: 0.0342 - val_accuracy: 0.9906
    Epoch 10/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0591 - accuracy: 0.9792 - val_loss: 0.0255 - val_accuracy: 0.9951
    Epoch 11/25
    15714/15714 [==============================] - 63s 4ms/step - loss: 0.0607 - accuracy: 0.9799 - val_loss: 0.0279 - val_accuracy: 0.9933
    Epoch 12/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0535 - accuracy: 0.9819 - val_loss: 0.0261 - val_accuracy: 0.9915
    Epoch 13/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0565 - accuracy: 0.9796 - val_loss: 0.0269 - val_accuracy: 0.9915
    Epoch 14/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0529 - accuracy: 0.9798 - val_loss: 0.0261 - val_accuracy: 0.9924
    Epoch 15/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0521 - accuracy: 0.9808 - val_loss: 0.0171 - val_accuracy: 0.9958
    Epoch 16/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0527 - accuracy: 0.9810 - val_loss: 0.0228 - val_accuracy: 0.9940
    Epoch 17/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0499 - accuracy: 0.9821 - val_loss: 0.0370 - val_accuracy: 0.9895
    Epoch 18/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0510 - accuracy: 0.9819 - val_loss: 0.0234 - val_accuracy: 0.9944
    Epoch 19/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0488 - accuracy: 0.9828 - val_loss: 0.0272 - val_accuracy: 0.9944
    Epoch 20/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0469 - accuracy: 0.9814 - val_loss: 0.0213 - val_accuracy: 0.9955
    Epoch 21/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0441 - accuracy: 0.9849 - val_loss: 0.0199 - val_accuracy: 0.9942
    Epoch 22/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0496 - accuracy: 0.9821 - val_loss: 0.0223 - val_accuracy: 0.9955
    Epoch 23/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0469 - accuracy: 0.9846 - val_loss: 0.0198 - val_accuracy: 0.9947
    Epoch 24/25
    15714/15714 [==============================] - 63s 4ms/step - loss: 0.0432 - accuracy: 0.9831 - val_loss: 0.0246 - val_accuracy: 0.9938
    Epoch 25/25
    15714/15714 [==============================] - 62s 4ms/step - loss: 0.0446 - accuracy: 0.9829 - val_loss: 0.0191 - val_accuracy: 0.9960

```python
# plot training and validation accuracy across epochs
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```
![fake_news_hw_29_1.png](/images/fake_news_hw_29_1.png)
    
In my third model, I will use **both the article title and the article text** as input.

```python
# vectorizing input for both title and text
title_features = vectorize_layer(title_input)
text_features = vectorize_layer(text_input)

# create a shared embedding layer for title and text
shared_embedding = layers.Embedding(size_vocabulary, 3, name = "embedding")
title_features = shared_embedding(title_features)
text_features = shared_embedding(text_features)

# add different layers for both features
title_features = layers.Dropout(0.2)(title_features)
text_features = layers.Dropout(0.2)(text_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
title_features = layers.Dropout(0.2)(title_features)
text_features = layers.Dropout(0.2)(text_features)
title_features = layers.Dense(32, activation='relu')(title_features)
text_features = layers.Dense(32, activation='relu')(text_features)

# concatenate title and text features
features = layers.concatenate([title_features, text_features], axis = 1)
features = layers.Dense(32, activation='relu')(features)

output = layers.Dense(2, name = "fake")(features)
```

```python
model3 = keras.Model(
    inputs = [title_input, text_input],
    outputs = output
)

model3.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)

history = model3.fit(train, 
                     validation_data=val,
                     epochs = 25)
```

    Epoch 1/25
    15714/15714 [==============================] - 79s 5ms/step - loss: 0.1912 - accuracy: 0.9173 - val_loss: 0.1269 - val_accuracy: 0.9517
    Epoch 2/25
    15714/15714 [==============================] - 78s 5ms/step - loss: 0.1091 - accuracy: 0.9621 - val_loss: 0.0682 - val_accuracy: 0.9797
    Epoch 3/25
    15714/15714 [==============================] - 78s 5ms/step - loss: 0.0883 - accuracy: 0.9689 - val_loss: 0.0549 - val_accuracy: 0.9866
    Epoch 4/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0864 - accuracy: 0.9700 - val_loss: 0.0560 - val_accuracy: 0.9877
    Epoch 5/25
    15714/15714 [==============================] - 78s 5ms/step - loss: 0.0775 - accuracy: 0.9717 - val_loss: 0.0604 - val_accuracy: 0.9779
    Epoch 6/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0709 - accuracy: 0.9749 - val_loss: 0.0473 - val_accuracy: 0.9880
    Epoch 7/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0675 - accuracy: 0.9767 - val_loss: 0.0317 - val_accuracy: 0.9922
    Epoch 8/25
    15714/15714 [==============================] - 78s 5ms/step - loss: 0.0666 - accuracy: 0.9749 - val_loss: 0.0722 - val_accuracy: 0.9646
    Epoch 9/25
    15714/15714 [==============================] - 78s 5ms/step - loss: 0.0631 - accuracy: 0.9779 - val_loss: 0.0296 - val_accuracy: 0.9920
    Epoch 10/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0576 - accuracy: 0.9778 - val_loss: 0.0339 - val_accuracy: 0.9929
    Epoch 11/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0608 - accuracy: 0.9783 - val_loss: 0.0236 - val_accuracy: 0.9942
    Epoch 12/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0573 - accuracy: 0.9786 - val_loss: 0.0263 - val_accuracy: 0.9924
    Epoch 13/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0553 - accuracy: 0.9805 - val_loss: 0.0309 - val_accuracy: 0.9904
    Epoch 14/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0542 - accuracy: 0.9815 - val_loss: 0.0253 - val_accuracy: 0.9949
    Epoch 15/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0517 - accuracy: 0.9814 - val_loss: 0.0226 - val_accuracy: 0.9938
    Epoch 16/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0531 - accuracy: 0.9809 - val_loss: 0.0272 - val_accuracy: 0.9933
    Epoch 17/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0512 - accuracy: 0.9821 - val_loss: 0.0277 - val_accuracy: 0.9929
    Epoch 18/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0456 - accuracy: 0.9844 - val_loss: 0.0250 - val_accuracy: 0.9938
    Epoch 19/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0443 - accuracy: 0.9845 - val_loss: 0.0518 - val_accuracy: 0.9813
    Epoch 20/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0460 - accuracy: 0.9852 - val_loss: 0.0188 - val_accuracy: 0.9938
    Epoch 21/25
    15714/15714 [==============================] - 78s 5ms/step - loss: 0.0444 - accuracy: 0.9856 - val_loss: 0.0210 - val_accuracy: 0.9951
    Epoch 22/25
    15714/15714 [==============================] - 78s 5ms/step - loss: 0.0465 - accuracy: 0.9837 - val_loss: 0.0215 - val_accuracy: 0.9947
    Epoch 23/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0410 - accuracy: 0.9856 - val_loss: 0.0301 - val_accuracy: 0.9909
    Epoch 24/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0404 - accuracy: 0.9863 - val_loss: 0.0239 - val_accuracy: 0.9940
    Epoch 25/25
    15714/15714 [==============================] - 77s 5ms/step - loss: 0.0416 - accuracy: 0.9876 - val_loss: 0.0198 - val_accuracy: 0.9955

```python
# plot training and validation accuracy across epochs
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```  
![fake_news_hw_33_1.png](/images/fake_news_hw_33_1.png)
    
After analyzing the performance of my models, we see that all three models were successful. It is clear that the model that uses just **article text** and the model that uses both **article title and article text** are most successful. However, it appears that overfitting could be more of an issue with the model that solely used **article text**. Thus, algorithms should use both title and text when seeking to detect fake news.

## Evaluating the Final Model

```python
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true"
test_df = pd.read_csv(test_url)
test_data = make_dataset(test_df)

# evaluate the best model on the test dataset
model3.evaluate(test_data)
```

    22449/22449 [==============================] - 60s 3ms/step - loss: 0.0611 - accuracy: 0.9814





    [0.06108703091740608, 0.981380045413971]

My final model which includes both the **article title and article text** has 98.14% accuracy on the test data. Thus, if we used my model as a fake news detector, it would be right approximately 98.14% of the time.

## Creating an Embedding Visualization

```python
weights = model3.get_layer('embedding').get_weights()[0] # get the weights from the embedding layer
vocab = vectorize_layer.get_vocabulary()                # get the vocabulary from our data prep for later

pca = PCA(n_components=2)
weights = pca.fit_transform(weights)

embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})

fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = list(np.ones(len(embedding_df))),
                 size_max = 10,
                 hover_name = "word")

fig.show()
```
{% include embedding_vis.html %}

Based on my embedding visualization, we can see that `conservative`, `economic`, `political`, `leader`, and `federal` all lie closely to one another towards the center of the plot. This makes sense because all of the words are related to the government and the overall status and views of people.