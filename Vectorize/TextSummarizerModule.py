from transformers import pipeline


class TextSummarizer():
    def __init__(self, model = "Falconsai/text_summarization", pipelineName = "summarization", max_length = 300, min_length = 75):
        self.model = model
        self.pipelineName = pipelineName
        self.max_length = max_length
        self.min_length = min_length
        self.summarizer = pipeline(self.pipelineName, model=self.model)

    def summarize(self, text):
        
        return self.summarizer(text, min_length=self.min_length, max_length=self.max_length, do_sample=False)[0]['summary_text']
    
    def summarizeMany(self, texts):
        summaries = []
        for text in texts:
            summaries.append(self.summarize(text))
        return summaries


def main():
    TS = TextSummarizer()
    
    article = "The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published an article titled 'Computing Machinery and Intelligence' which proposed what is now called the Turing test as a criterion of intelligence. The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. The authors claimed that within three or five years, machine translation would be a solved problem. However, real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced. Little further research in machine translation was conducted until the late 1980s when the first statistical machine translation systems were developed."
    
    print("Original Length:", len(article))
    print("Summary Length:", len(TS.summarize(article)))

if __name__ == '__main__':
    main()
