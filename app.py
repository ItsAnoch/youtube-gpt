from py_youtube import Search, Data
from youtube_transcript_api import YouTubeTranscriptApi 
import openai
from transformers import pipeline


# Configs
transcript = ''
transcriptWordLimit = 1000 # 0 for no word limit.
maxSummaryLength = 1000 # Maximum summary length (Set to 0 for no max)
useBetterSummary = True # Can summarize longer videos, if set to True then maxSummaryLength can be set to 0
apiKey = "[ YOUR OPENAI KEY ]" # Your OpenAI key.

# --------------------------------------------------------------------------------

if useBetterSummary:
    print("\nLoading second summarizer..")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Done loading!\n\n")

videoQuery = input("Enter a video URL, ID or title. ")

# Finding the video oj YouTube
video = Search(videoQuery, limit=1).videos()[0]

print(f"\n\nFound video: {video['title']}")

# Getting the video transcript
try:
    rawVideoTranscript = YouTubeTranscriptApi.get_transcript(video['id'])
    text = [d['text'] for d in rawVideoTranscript]
    transcript = ''.join(text)

    if transcriptWordLimit > 0:
        transcript = " ".join(transcript.split(" ")[:transcriptWordLimit])

except Exception as e:
    print("Couldn't transcribe video. Ending.")
    print(e)

# String Chunk Convertor
# Credit: Nicholas Renotte
def textToChunks(string: str, maxChunkSize=500):
    STRING = string.replace('.', '.<eos>')
    STRING = string.replace('!', '!<eos>')
    STRING = string.replace('?', '?<eos>')
    sentences = string.split('<eos>')

    currentChunk = 0
    chunks = []

    for sentence in sentences:
        if len(chunks) == currentChunk+1:
            if len(chunks[currentChunk] + len(sentence.split(' '))) <= maxChunkSize:
                chunks[currentChunk].extend(sentence.split(' '))
            else:
                currentChunk += 1
                chunks.append(sentence.split(' '))
        
        else:
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = " ".join(chunks[chunk_id])

    return chunks

if transcript:
    # Setting OpenAI key
    openai.api_key = apiKey

    # Getting video data to use for query
    videoID = video['id']
    videoData = Data(f'https://youtu.be/{videoID}').data()    

    if useBetterSummary:
        chunks = textToChunks(transcript)
        try:
            transcript = summarizer(chunks, max_length=maxSummaryLength, min_length=int(maxSummaryLength/2), do_sample=False)
        except:
            print(f'\n\nVideo is too long to summarize. Shorted to {transcriptWordLimit} words.')

    
    if transcriptWordLimit > 0:
        transcript = " ".join(transcript.split(" ")[:transcriptWordLimit])

    AIContext = f"Video Title: {videoData['title']}\nVideo Creator: {videoData['channel_name']}"
    AIPrompt = f"Summarize the following: {transcript}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": AIContext},
            {"role": "user", "content": AIPrompt},
        ],
        max_tokens=1000, # Max Free User Token
        top_p=1, 
        frequency_penalty=2,
        presence_penalty=2,
    )

    transcriptSummary = response.choices[0].message.content#.replace('\n', '<br><br>')
    print(f"\n\n{transcriptSummary}\n\n")
