import gradio as gr
from utils import *


def get_result(text):
    
    path_to_pretrained_model = '../model6'
    model = BertForSequenceClassification.from_pretrained(path_to_pretrained_model)
    tokenizer = BertTokenizerFast.from_pretrained('sberbank-ai/ruBert-base')
    label_df = pd.read_csv('../label_df.csv')
    
    predicted_category = bert_inference(text, tokenizer, model, label_df)
    
    return predicted_category


iface = gr.Interface(
    get_result,
    "text",
    "text",
    title="Предсказать категорию",
)

if __name__ == "__main__":
    iface.launch()