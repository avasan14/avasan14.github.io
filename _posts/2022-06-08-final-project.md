---
layout: post
title: Performing an Analysis on the Written Opinions of Supreme Court Justices
---

This is my final project, in which my group utilized `tensorflow` and NLP techniques in order to perform a sentiment analysis on the written opinions of Supreme Court Justices.

Here is a link to our GitHub repository that contains all of our code: https://github.com/RaymondBai/PIC16B-Project

## Overview 
For our project, we conducted a sentiment analysis on the opinions of Supreme Court Justices with the aim to differentiate and highlight the unique "legal writing styles" of the Justices, which is beneficial for people learning about legal writing and may reveal Justices' legal philosophy. Our methodology included downloading large volumes of Supreme Court opinion PDFs from the official website. Then, we used OCR tools to detect and store the text before using regular expressions to separate the opinions and identify the author in order to construct our official dataset CSV. After preparing the data, we utilized NLP packages and tensorflow in order to find high prevalence words for each opinion type and author, as well as score the overall sentiment in the opinion. Once we created our models for both type and author classification based on the text, we tested these models on completely unseen data from the past 2 months. After examining our results, which were poor on the unseen data, we attempted to recreate our models after removing the justices from the training set who were not seen in the test set. As a result, our results seemed to improve. 

Here is a flowchart depicting our steps taken throughout this process:
![project_flowchart.png](/images/project_flowchart.png)

## Technical Components

These are some key technical components that I will explain in further detail.

1. Web Scraping
2. OCR/Text Cleaning
3. Type and Author Classification: Training the data and Evaluating on Unseen data

### Web Scraping
We began by scraping the Supreme Court Opinions Directory which contained pdf links of the Supreme Court opinions from 2021 to 2014. To create the scraper, we made a parse method that used the relevant css selectors and tags to acquire the opinion PDF links for each month of the year. Next we utilized a for loop to index through the list of PDF links and download the PDFs. A second parse method was created to go to the website links of each year and scrape and continue this process of downloading the PDFs. 

In the settings file, we specified “pdf” to be the document format to save the files as. A download delay was also implemented. Without this, multiple threads will try to write to the csv file at the same time and this will produce a file lock error in the command prompt.

Here is a screenshot of the website that we scraped our data from along with our spider code:
![scotus_website.png](/images/scotus_website.png)

```python
import scrapy
from pic16bproject.items import Pic16BprojectItem

# Spider for downloading preliminary prints
class courtscraper(scrapy.Spider):
     name = 'court_spider'
    
    start_urls = ['https://www.supremecourt.gov/opinions/USReports.aspx']

    def parse(self, response):

         pdfs = [a.attrib["href"] for a in response.css("div#accordion2 a")]
         prefix = "https://www.supremecourt.gov/opinions/"
         pdfs_urls = [prefix + suffix for suffix in pdfs]


         for url in pdfs_urls:
            item = Pic16BprojectItem() #define it items.py
            item['file_urls'] = [url]
            yield item

# items.py file
class Pic16BprojectItem(scrapy.Item):
    file_urls = scrapy.Field()
    files = scrapy.Field()

#settings.py file

BOT_NAME = 'pic16bproject'

SPIDER_MODULES = ['pic16bproject.spiders']
NEWSPIDER_MODULE = 'pic16bproject.spiders'

ITEM_PIPELINES = {
    'scrapy.pipelines.files.FilesPipeline' : 1,
}

FILES_STORE = "pdf"
FILES_RESULT_FIELD = 'files'

ROBOTSTXT_OBEY = True
```

### OCR and Text Cleaning

Once we completely scraped the website for our data, we performed OCR and Text Cleaning in order to create a CSV file that could be used for our analysis. Our goal was to create a file with three columns: `Author`, `Type`, `Text`. 
1. `Author`: The name of a Supreme Court Justice
2. `Type`: The opinion type
3. `Text`: The cleaned text for the corresponding justice's opinion

Our methodology involved multiple parts. First, we iterated through all the pages in an opinion and stored it as a jpg file. We declared the filename for each page of PDF as JPG and then saved the image of the page in our system. Then, we incremented the counter to update filename accordingly. As we progressed, we added code within this same for loop in order to continue iterating over the pages. 

For part 2, we created a text file to write the output. We did this by opening the file in append mode, then iterating from 1 to the total number of pages. Then, for each page, we recognized the text as a string in image using pytesseract. If the page is a syllabus page or not an opinion page, we skipped and removed the file as there no need to append the text. We made sure to then note down the page was skipped, removed the image, and moved on to next page. Then, we restored sentences by using regex to clean our text further by removing headers and boundaries, and separating the opinions. Finally, we wrote to the text file and removed the image. 

For part 3, we read in the text file and converted the text to our desired data frame. Here is the code for our OCR and text cleaning as well as what the first 10 rows of our final, cleaned data looks like:
```python
# For every opinion PDF (donwloaded from spider)
for op in [i for i in os.listdir("./opinion_PDFs") if i[-3:] == 'pdf']:
    
    # *** Part 1 ***
    pages = convert_from_path("./opinion_PDFs/" + op, dpi = 300)
    image_counter = 1
    # Iterate through all the pages in this opinion and store as jpg
    for page in pages:
        # Declaring filename for each page of PDF as JPG
        # For each page, filename will be:
        # PDF page 1 -> page_1.jpg
        # ....
        # PDF page n -> page_n.jpg
        filename = "page_"+str(image_counter)+".jpg"
        # Save the image of the page in system
        page.save(filename, 'JPEG')
        # Increment the counter to update filename
        image_counter = image_counter + 1
    image_counter = image_counter - 1
    
    # *** Part 2 ***
    # Creating a text file to write the output
    outfile = "./opinion_txt/" + op.split(".")[0] + "_OCR.txt"
    # Open the file in append mode
    f = open(outfile, "w")
    
    # Iterate from 1 to total number of pages
    skipped_pages = []
    print("Starting OCR for " + re.findall('([0-9a-z-]+)_', op)[0])
    print("Reading page:")
    for i in range(1, image_counter + 1):
        print(str(i) + "...") if i==1 or i%10==0 or i==image_counter else None
        # Set filename to recognize text from
        filename = "page_" + str(i) + ".jpg"
        # Recognize the text as string in image using pytesserct
        text = pytesseract.image_to_string(Image.open(filename))
        # If the page is a syllabus page or not an opinion page
        # marked by "Opinion of the Court" or "Last_Name, J. dissenting/concurring"
        # skip and remove this file; no need to append text
        is_syllabus = re.search('Syllabus\n', text) is not None
        is_maj_op = re.search('Opinion of [A-Za-z., ]+\n', text) is not None
        is_dissent_concur_op = re.search('[A-Z]+, (C. )?J., (concurring|dissenting)?( in judgment)?', text) is not None
        if is_syllabus or ((not is_maj_op) and (not is_dissent_concur_op)):
            # note down the page was skipped, remove image, and move on to next page
            skipped_pages.append(i)
            os.remove(filename)
            continue
        # Restore sentences
        text = text.replace('-\n', '')
        # Roman numerals header
        text = re.sub('[\n]+[A-Za-z]{1,4}\n', '', text)
        # Remove headers
        text = re.sub("[\n]+SUPREME COURT OF THE UNITED STATES[\nA-Za-z0-9!'#%&()*+,-.\/\[\]:;<=>?@^_{|}~—’ ]+\[[A-Z][a-z]+ [0-9]+, [0-9]+\][\n]+",
                  ' ', text)
        text = re.sub('[^\n]((CHIEF )?JUSTICE ([A-Z]+)[-A-Za-z0-9 ,—\n]+)\.[* ]?[\n]{2}',
                  '!OP START!\\3!!!\\1!!!', text)
        text = re.sub('[\n]+', ' ', text) # Get rid of new lines and paragraphs
        text = re.sub('NOTICE: This opinion is subject to formal revision before publication in the preliminary print of the United States Reports. Readers are requested to noti[f]?y the Reporter of Decisions, Supreme Court of the United States, Washington, D.[ ]?C. [0-9]{5}, of any typographical or other formal errors, in order that corrections may be made before the preliminary print goes to press[\.]?',
                      '', text)
        text = re.sub('Cite as: [0-9]+[ ]?U.S.[_]* \([0-9]+\) ([0-9a-z ]+)?(Opinion of the Court )?([A-Z]+,( C.)? J., [a-z]+[ ]?)?',
                      '', text)
        text = re.sub(' JUSTICE [A-Z]+ took no part in the consideration or decision of this case[\.]?', '', text)
        text = re.sub('[0-9]+ [A-Z!&\'(),-.:; ]+ v. [A-Z!&\'(),-.:; ]+ (Opinion of the Court )?(dissenting[ ]?|concurring[ ]?)?',
                  '', text)
        # Remove * boundaries
        text = re.sub('([*][ ]?)+', '', text)
        # Eliminate "It is so ordered" after every majority opinion
        text = re.sub(' It is so ordered\. ', '', text)
        # Eliminate opinion header
        text = re.sub('Opinion of [A-Z]+, [C. ]?J[\.]?', '', text)
        # Separate opinions
        text = re.sub('!OP START!', '\n', text)
    
        # Write to text
        f.write(text)
    
        # After everything is done for the page, remove the page image
        os.remove(filename)
    # Close connection to .txt file after finishing writing
    f.close()
    
    # Now read in the newly created txt file as a pandas data frame if possible
    
    try:
        op_df = pd.read_csv("./opinion_txt/" + op.split(".")[0] + "_OCR.txt",
                            sep = re.escape("!!!"), engine = "python",
                            names = ["Author", "Header", "Text"])
        op_df.insert(1, "Docket_Number", re.findall("([-a-z0-9 ]+)_", op)[0])
        op_df["Type"] = op_df.Header.apply(opinion_classifier)
        
        # Lastly add all the opinion info to the main data frame
        opinion_df = opinion_df.append(op_df, ignore_index = True)
        os.remove("./opinion_PDFs/" + op)
        print("Task completed\nPages skipped: " + str(skipped_pages) + "\n")
    except:
        print("Error in CSV conversion. Pages NOT added!\n")
        
print("-----------------------\nAll assigned OCR Completed")
```

  <div id="df-2884948c-c120-4f0a-b532-3df596371ce8">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Author</th>
      <th>Text</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GINSBURG</td>
      <td>In the Federal Employees Health Benefits Act o...</td>
      <td>Opinion</td>
    </tr>
    <tr>
      <th>1</th>
      <td>THOMAS</td>
      <td>I join the opinion of the Court with one reser...</td>
      <td>Concurrence</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ROBERTS</td>
      <td>In our judicial system, “the public has a righ...</td>
      <td>Opinion</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KAVANAUGH</td>
      <td>The Court today unanimously concludes that a P...</td>
      <td>Concurrence in Judgment</td>
    </tr>
    <tr>
      <th>4</th>
      <td>THOMAS</td>
      <td>Respondent Cyrus Vance, Jr., the district atto...</td>
      <td>Dissent</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ALITO</td>
      <td>This case is almost certain to be portrayed as...</td>
      <td>Dissent</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KENNEDY</td>
      <td>The classic example of a property taking by th...</td>
      <td>Opinion</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ROBERTS</td>
      <td>The Murr family owns two adjacent lots along t...</td>
      <td>Dissent</td>
    </tr>
    <tr>
      <th>8</th>
      <td>THOMAS</td>
      <td>I join THE CHIEF JUSTICE’s dissent because it ...</td>
      <td>Dissent</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BREYER</td>
      <td>The Centers for Disease Control and Prevention...</td>
      <td>Dissent</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2884948c-c120-4f0a-b532-3df596371ce8')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2884948c-c120-4f0a-b532-3df596371ce8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2884948c-c120-4f0a-b532-3df596371ce8');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

### Type and Author Classification
 
We used tensorflow in order to classify all of the opinion types and justices, labeled as authors, based on the text alone. To do this, we created two data frames: one with type and text as the columns, and another with author and text as the columns. Then, we converted each type and column into integer labels using a label encoder in order to move forward with our classification task. We split our data into 70% training, 10% validation, and 20% testing in order to train our models and compare our resulting accuracies. We made sure to vectorize the data as well in order to properly train and test our models. Both the type and author models implemented a sequential model that used an embedding layer, two dropout layers, a 1D global average pooling layer, and a dense layer. The dimensions for the output and dense layer were altered based on the total number of opinion types (4) and total number of authors (12). We experienced great success with the training and validation accuracies for both models. For the type model, the training accuracies hovered around 92% and the validation accuracies settled around 99% as the later epochs were completed. For the author model, the training accuracies hovered around 87% and the validation accuracies settled around 97% as the later epochs were completed. Further, we did not worry too much about overfitting as there was a large amount of overlap between training and validation accuracies and there was never too much of a dropoff between the two. After training our models, we evaluated them on the testing set which was the random 20% of the original data. Once again, experienced great success as the type and author test accuracies were approximately 99.5% and 95.6%, respectively. Thus, our models performed very well on the original dataset. Here is what the code for our type model looked like. The author model is very similar, but uses the author data instead.

```python
max_tokens = 2000
sequence_length = 25 

vectorize_layer = TextVectorization(
    standardize =  standardization, 
    output_mode = 'int', 
    max_tokens = max_tokens, 
    output_sequence_length =  sequence_length
)

opinion_type = type_train.map(lambda x, y: x)
vectorize_layer.adapt(opinion_type)

def vectorize_pred(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), [label]

train_vec = type_train.map(vectorize_pred)
val_vec = type_val.map(vectorize_pred)
test_vec = type_test.map(vectorize_pred)

type_model = tf.keras.Sequential([
    layers.Embedding(max_tokens, output_dim = 4, name = "embedding"),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(), 
    layers.Dropout(0.2), 
    layers.Dense(4)
])

type_model.compile(loss = losses.SparseCategoricalCrossentropy(from_logits = True),
                   optimizer = "adam", 
                   metrics = ["accuracy"])

history = type_model.fit(train_vec, epochs = 80, validation_data = val_vec)

type_model.evaluate(test_vec)
```

We also created an additional testing set that included unseen data from the past two months alone. This is where our models seemed to falter. Specifically, our type model testing accuracy was approximately 28.1% and our author model testing accuracy was approximately 3.1%. These are clearly much lower than the testing accuracies from our initial data. Thus, we performed further evaluation of our datasets and noticed some variations. Specifically, the unseen test set which has all the data from the last two months, consisted of fewer authors than our original data. So, we removed the justices from the original dataset who were not seen in the data from the last two months and retrained and tested our models once again. Similar to the first time, the training and validation accuracies were very high. However, we did notice a slight increase in our testing accuracies as the type model improved to approximately 34.4% and our author model improved to approximately 15.6%. Although these are still rather low, we believe that further inspection into our test dataset would provide us with more clarity about potential improvements that we could make to our model so that it performs better with the testing data. Here is a table of the actual opinion types as well as our predicted types.

  <div id="df-6d5e42a3-5e4f-435f-8e10-de185f61cde1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted</th>
      <th>Actual</th>
      <th>Correct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Opinion</td>
      <td>Opinion</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dissent</td>
      <td>Dissent</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dissent</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dissent</td>
      <td>Concurrence</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dissent</td>
      <td>Concurrence in Judgment</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Dissent</td>
      <td>Concurrence in Judgment</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Concurrence</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Dissent</td>
      <td>Dissent</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Dissent</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dissent</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Dissent</td>
      <td>Concurrence</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Dissent</td>
      <td>Dissent</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Concurrence</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Dissent</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Opinion</td>
      <td>Dissent</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Dissent</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Dissent</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Opinion</td>
      <td>Dissent</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Dissent</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Opinion</td>
      <td>Opinion</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Dissent</td>
      <td>Concurrence</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Opinion</td>
      <td>Concurrence</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Dissent</td>
      <td>Dissent</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Dissent</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Dissent</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Concurrence</td>
      <td>Concurrence</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Dissent</td>
      <td>Dissent</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Dissent</td>
      <td>Dissent</td>
      <td>True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Dissent</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Opinion</td>
      <td>Dissent</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Dissent</td>
      <td>Opinion</td>
      <td>False</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Concurrence</td>
      <td>Dissent</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6d5e42a3-5e4f-435f-8e10-de185f61cde1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-6d5e42a3-5e4f-435f-8e10-de185f61cde1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6d5e42a3-5e4f-435f-8e10-de185f61cde1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>

## Concluding Remarks
As mentioned earlier, with more time, we could definitely identify potential solutions to improving our accuracy on the unseen testing set for both the type and author models. However, we believe that our testing results from the original dataset shows us that the models are performing well. Thus, finding a way to normalize the datasets to ensure that the text, type, and author are completely reliable, we could be able to perform an analysis on unseen data that is more consistent with our initial results. Overall, we feel that our intended deliverables were reached as we now have a better understanding of the writing styles of the Supreme Court Justices.