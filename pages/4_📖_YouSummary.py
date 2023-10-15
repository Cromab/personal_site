import streamlit as st
from pytube import extract
from youtube_transcript_api import YouTubeTranscriptApi
from time import sleep
import re
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from heapq import nlargest

#--- Preamble ---#
#YouStudy page configuration
st.set_page_config(page_title="YouStudy", page_icon=":open_book:", layout="wide")

#Use local css
def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

with st.container():
    st.markdown("""
                Have you ever watched a long Youtube Tutorial and wanted a quick reference? If you're anything like me, you like to take notes.
                And if you're anything like me, you take way too many notes. Instead of that, what if we leverage some NLP methods to summarize text?
                Presenting Youtorial...or Yoututorial...or maybe YouStudy? I don't know what name sounds good. YouLearn? Maybe YouSummary?
                ##
                To test this I've used my favorite piece of youtube media, and probably the greatest piece of performance art of all time, 
                featuring John Blythe Barrymore, brother of Drew Barrymore. Please ignore the unhinged summary, the video is also unhinged.
                """)

#--- Take Input for url and return transcript of video ---#
with st.form("yt_url"):
        submitted = st.form_submit_button(label="Generate Study Guide")
        url = st.text_input('Youtube URL', 'https://www.youtube.com/watch?v=NAh9oLs67Cw')
        video_id = extract.video_id(url)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id) 
            transcript = " ".join(item["text"] for item in transcript)
        except:
            "Transcript not enabled for this video. Please enter url of a captioned video."
            st.stop()
        
if submitted:
        #Check text field is not empty
        if not url.strip():
                st.error('Warning: Please enter text')
        else:
                with st.spinner(text = f'Extracting information...'):
                        sleep(3)


#--- Tokenization and Data clean up ---#
#Stopwords and punctuation additions
nltk.download("stopwords")
nltk.download("punkt")
stop_words = stopwords.words('english')
punctuation = punctuation + '\n' + "\'"
transcript = re.sub(f"[!]", '', transcript)
transcript = re.sub(f"\[.*\]", '', transcript)

#Frequency Table Creation
tokens = word_tokenize(transcript)
word_frequencies = {}
for word in tokens:
    if word.lower() not in stop_words:
        if word.lower() not in punctuation:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
#Frequency Table division by max word count
max_frequency = max(word_frequencies.values())
for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency

#Tokenize transcript into 'sentences'
sentences = sent_tokenize(transcript)


#--- Find weighted frequencies of 'sentences' ---#
#Score sentences by its words frequency
sentence_weight = {}
for sentence in sentences:
    sentence_wordcount = (len(word_tokenize(sentence)))
    sentence_wordcount_without_stop_words = 0
    for word_weight in word_frequencies:
        if word_weight in sentence.lower():
            sentence_wordcount_without_stop_words += 1
            if sentence in sentence_weight:
                sentence_weight[sentence] += word_frequencies[word_weight]
            else:
                sentence_weight[sentence] = word_frequencies[word_weight]
    #sentence_weight[sentence] = sentence_weight[sentence]

#--- Summarize Text ---#
select_length = int(len(sentence_weight)*0.3)
summary = nlargest(select_length, sentence_weight, key = sentence_weight.get)
final_summary = [word for word in summary]
summary = ' '.join(final_summary)
with st.container():
    st.write("Summary of your video:")
    st.write("---")
    st.write(summary)
    st.write("---")
    
#--- Closing Statements ---#
st.markdown("""
            Text Summarization is **very** important. It involves a lot of general data clean up and honing of important words.
            More importantly though, it's a good building block for more complex programatic tasks. While summarizations above (for those curious
            we used a very straight algorithm based on the frequency of words to give each word and sentence a 'weight') are neat, something even
            cooler, for example, could be the creation of a quiz using this summary. Questgen's api is great place to start if this sounds interesting,
            or if you prefer to leverage a LLM like Llama2 or ChatGPT, they are more than up to the task. 
            """)
