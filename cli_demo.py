from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
import os
import nltk
from models.loader.args import parser
import models.shared as shared
from models.loader import LoaderCheckPoint
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# Show reply with source text from input document
REPLY_WITH_SOURCE = True


def main():

    llm_model_ins = shared.loaderLLM()
    llm_model_ins.history_len = LLM_HISTORY_LEN

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model=llm_model_ins,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)
    vs_path = None
    while not vs_path:
        filepath = input("Input your local knowledge file path 请输入本地知识文件路径：")
        # 判断 filepath 是否为空，如果为空的话，重新让用户输入,防止用户误触回车
        if not filepath:
            continue
        vs_path, _ = local_doc_qa.init_knowledge_vector_store(filepath)
    history = []

    # Read input text from a file
    input_text = """7星彩的基本投注规则是什么？
在单场大小分游戏中，如果比赛进行中因故中断了怎么办？
什么情况下会认定因故中断的比赛为无效场次？
单场大小分游戏中，比赛开始后。参赛队伍和我投注时不同，投注还有效吗？
大乐透一等奖的奖金有多少？
大乐透一等奖奖金如何计算？
大乐透一等奖奖金具体计算方法是什么？
单场竞猜胜分差游戏中的“主26+”是什么？
7星彩各个奖项的中奖条件是怎样的？
7星彩的三等奖中奖条件是什么？
单场胜分差游戏怎么玩？
大乐透怎么玩？
什么是过关投注？
过关投注的规则是什么？
我想买足彩，有哪些玩法？
让分胜负游戏的规则是什么？
7星彩是什么？它和大乐透有什么不同？
中国篮球彩票的单场竞猜大小分游戏是什么意思？我如何参与并下注？
单场竞猜让分胜负游戏中的"让分"是什么含义？
我如何在单场竞猜大小分游戏中下注？
单场竞猜胜分差游戏的规则是怎样的？
单场竞猜胜分差游戏设置哪些投注选项？
中奖后我该如何领取彩票奖金？
中奖后我该如何兑奖？
单场竞猜大小分游戏中，大小分的设定是由谁决定的？
单场竞猜大小分游戏奖金是如何设置的？
篮球彩票单场竞猜大小分游戏中的“大小分”是什么意思？
在单场竞猜胜分差游戏中，如果比赛最后结果与我的预测不完全相符，是否还有机会中奖？
如果我购买的彩票遗失了或者损坏了，还能兑奖吗？
大乐透和七星彩分别什么时候开奖？
大乐透什么时候开奖？
七星彩什么时候开奖？
    """

    # Split input_text into paragraphs
    paragraphs = input_text.strip().split('\n')

    for query in paragraphs:
        last_print_len = 0
        for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
                                                                     vs_path=vs_path,
                                                                     chat_history=[],
                                                                     streaming=STREAMING):
            if STREAMING:
                print(resp["result"][last_print_len:], end="", flush=True)
                last_print_len = len(resp["result"])
            else:
                print(resp["result"])
        if REPLY_WITH_SOURCE:
            source_text = [f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
                           # f"""相关度：{doc.metadata['score']}\n\n"""
                           for inum, doc in
                           enumerate(resp["source_documents"])]
            # print("\n\n" + "\n\n".join(source_text))


if __name__ == "__main__":
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    main()
