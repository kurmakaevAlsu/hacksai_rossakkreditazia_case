import gradio as gr
from utils import *
from bert_utils import *

def get_result(file):
    data = pd.read_csv(file.name, dtype=str)
    df = get_prediction_final(data)
    df.to_excel("result.xlsx")
    return "result.xlsx"


iface = gr.Interface(
    get_result,
    "file",
    "file",
    title='Проверить на ошибки присвоения категории',
)

if __name__ == "__main__":
    iface.launch()
