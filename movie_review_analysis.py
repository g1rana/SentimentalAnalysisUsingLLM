import os
import pandas as pd
import torch
import transformers


df_review = pd.read_csv("IMDB_Dataset.csv")
df_review.head(3)
df_review.iloc[0].review

transformers.utils.logging.set_verbosity_error()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline


phi3_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",torch_dtype="auto", trust_remote_code=True)
phi3_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

def zero_shot_promt(review):
    return """
            Classify the sentiment of the following review as  either 'positive' or 'negative'.
            Review: {review}.
            Sentiment:
            """.format(review= review)

def few_shot_prompt(review):
  return """
    Classify the sentiment of the follow movie review as either "positive" or "negative".

    Review: I had the pleasure of watching the Titanic movie. What a blast! Great story and cinematography.
    Sentiment:positive

    Review: The Room is a total waste of time. It's bad to the point of comical.
    Sentiment:negative

    Review: {review}.
    Sentiment:""".format(review=review)


def predict_sentiment(model, tokenizer,prompt):
   pipe = pipeline(
      "text-generation",
      model= model,
      tokenizer=tokenizer,
      device=device
   )
   generation_args = {
      "max_new_tokens" :1,
      "return_full_text": False,
      "do_sample": False,
   }
   message = [{"role":"user", "content": prompt}]
   chat_prompt = phi3_tokenizer.apply_chat_template(message,add_generation_prompt=True,tokenize=False)
   return pipe(chat_prompt,**generation_args)[0]['generated_text']




def zero_shot_sentiment_analysis(n=10):
   n_correct = 0
   for _, row in df_review.iloc[0:n].iterrows():
      review = row.review
      sentiment = row.sentiment
      prediction = predict_sentiment(phi3_model,phi3_tokenizer,zero_shot_promt(review))
      n_correct += prediction.lower().startwith(sentiment)
      accuracy = n_correct/n
      print(f'zero {accuracy=}')