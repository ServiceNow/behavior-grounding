import json
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import pickle as pkl
import numpy as np
import re
import ast




def urs_scoring_prompt(cfg: DictConfig, question: str, gpt4_answer: str, test_answer: str, intent: str):
    prompt_dict={"Leisure":'''请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是[休闲娱乐]，因此你需要从下面的5个维度对回答进行评估:
    1 满足用户需求(User Satisfaction)
    是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应
    2 趣味性 (Engagement)
    回答是否有趣、吸引人，帮助用户放松，提供了高质量的情绪价值或娱乐价值等
    3 适宜性 (Appropriateness)
    内容适宜所有用户，避免不当或冒犯性内容
    4 创造性 (Creativity)
    是否具有创新性或独特性，是否提供了新颖的见解或解决方法
    5 事实正确性 (Factuality)
    回答中提供的信息是否准确无误，是否基于可信的事实和数据
    注意，不要让回答的长度影响你的打分！回答不是越长越好，简洁并且满足上述要求的回答是好的。

    我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程：
    1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。
    2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个1～10的分数。
    3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
    4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。
    其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。

    当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；
    当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；
    当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；
    当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；
    只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。
    作为示例，参考答案可以得到8分。

    请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数： 
    {’维度一’: 打分, ’维度二’: 打分, ..., ’综合得分’: 打分}，例如：{{'满足用户需求': 6, '趣味性': 8..., '综合得分': 7}}。
    用户的提问：''',
    'Factual_QA':'''请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是[信息检索]，因此你需要从下面的5个维度对回答进行评估:
    1 事实正确性(Factuality)
    提供的信息是否准确无误，是否基于可信的事实和数据
    2 满足用户需求(User Satisfaction)
    是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应
    3 清晰度 (Clarity)
    回答是否表达清晰易懂，是否使用了简洁的语言和结构，以便用户可以轻松理解
    4 完备性 (Completeness)
    回答是否提供了足够的信息和细节，以满足用户的需求，是否遗漏了重要的方面
    5 逻辑连贯性(Logical Coherence)
    回答是否在整体上保持一致，是否在不同部分之间保持逻辑连贯性，避免了自相矛盾
    注意，不要让回答的长度影响你的打分！回答不是越长越好，简洁并且满足上述要求的回答是好的。

    我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程：
    1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。
    2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个1～10的分数。
    3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
    4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。
    其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。

    当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；
    当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；
    当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；
    当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；
    只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。
    作为示例，参考答案可以得到8分。

    请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数： 
    {’维度一’: 打分, ’维度二’: 打分, ..., ’综合得分’: 打分}，例如：{{'事实正确性': 6, '满足用户需求': 8..., '综合得分': 7}}。
    用户的提问：''',
    'Seek_Creativity':'''请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是[给出创意]，因此你需要从下面的5个维度对回答进行评估:
    1 满足用户需求(User Satisfaction)
    是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应
    2 逻辑连贯性(Logical Coherence)
    回答是否在整体上保持一致，是否在不同部分之间保持逻辑连贯性，避免了自相矛盾
    3 创造性(Creativity)
    是否具有创新性或独特性，是否提供了新颖的见解或解决方法
    4 丰富度(Richness)
    是否包含丰富的信息、深度、上下文考虑、多样性、详细解释和实例，以满足用户需求并提供全面理解
    5 事实正确性(Factuality)
    提供的信息是否准确无误，是否基于可信的事实和数据
    注意，不要让回答的长度影响你的打分！回答不是越长越好，简洁并且满足上述要求的回答是好的。

    我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程：
    1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。
    2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个1～10的分数。
    3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
    4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。
    其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。

    当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；
    当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；
    当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；
    当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；
    只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。
    作为示例，参考答案可以得到8分。

    请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数： 
    {’维度一’: 打分, ’维度二’: 打分, ..., ’综合得分’: 打分}，例如：{{'满足用户需求': 6, '逻辑连贯性': 8..., '综合得分': 7}}。
    用户的提问：''',
    'Ask_for_Advice':'''请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是[给出建议]，因此你需要从下面的5个维度对回答进行评估:
    1 满足用户需求(User Satisfaction)
    是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应
    2 事实正确性(Factuality)
    提供的信息是否准确无误，是否基于可信的事实和数据
    3 公平与可负责程度(Fairness and Responsibility)
    回答中提供的建议或信息是否可行，是否负有一定的责任，是否考虑了潜在风险和后果。
    4 创造性(Creativity)
    是否具有创新性或独特性，是否提供了新颖的见解或解决方法
    5 丰富度(Richness)
    是否包含丰富的信息、深度、上下文考虑、多样性、详细解释和实例，以满足用户需求并提供全面理解
    注意，不要让回答的长度影响你的打分！回答不是越长越好，简洁并且满足上述要求的回答是好的。

    我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程：
    1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。
    2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个1～10的分数。
    3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
    4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。
    其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。

    当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；
    当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；
    当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；
    当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；
    只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。
    作为示例，参考答案可以得到8分。

    请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数： 
    {’维度一’: 打分, ’维度二’: 打分, ..., ’综合得分’: 打分}，例如：{{'满足用户需求': 6, '事实正确性': 8..., '综合得分': 7}}。
    用户的提问：''',
    'Solve_Professional_Problem':'''请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是[解决专业问题]，因此你需要从下面的5个维度对回答进行评估: 
    1 事实正确性(Factuality)
    提供的信息是否准确无误，是否基于可信的事实和数据
    2 满足用户需求(User Satisfaction)
    是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应
    3 清晰度(Clarity)
    是否表达清晰易懂，是否使用了简洁的语言和结构，以便用户可以轻松理解
    4 逻辑连贯性(Logical Coherence)
    是否在整体上保持一致，是否在不同部分之间保持逻辑连贯性，避免了自相矛盾
    5 完备性(Completeness)
    回答是否提供了足够的信息和细节，以满足用户的需求，是否遗漏了重要的方面
    注意，回答不是越长越好，简短并且满足上述要求的回答是最好的。

    我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程：
    1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。
    2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个1～10的分数。
    3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
    4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。
    其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。

    当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；
    当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；
    当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；
    当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；
    只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分 的情况下，才能得9到10分。 
    作为示例，参考答案可以得到8分。

    请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整 数： 
    {’维度一’: 打分, ’维度二’: 打分, ..., ’综合得分’: 打分}，例如：{’事实正确性’: 9, ’满足用户需求’: 6, ’综合得分’: 7}。
    用户的提问：''',
    "API":'''请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是[解决专业问题]，因此你需要从下面的5个维度对回答进行评估: 
    1 事实正确性(Factuality)
    提供的信息是否准确无误，是否基于可信的事实和数据
    2 满足用户需求(User Satisfaction)
    是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应
    3 清晰度(Clarity)
    是否表达清晰易懂，是否使用了简洁的语言和结构，以便用户可以轻松理解
    4 逻辑连贯性(Logical Coherence)
    是否在整体上保持一致，是否在不同部分之间保持逻辑连贯性，避免了自相矛盾
    5 完备性(Completeness)
    回答是否提供了足够的信息和细节，以满足用户的需求，是否遗漏了重要的方面
    注意，回答不是越长越好，简短并且满足上述要求的回答是最好的。

    我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程：
    1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。
    2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个1～10的分数。
    3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
    4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。
    其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。

    当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；
    当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；
    当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；
    当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；
    只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分 的情况下，才能得9到10分。 
    作为示例，参考答案可以得到8分。

    请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整 数： 
    {’维度一’: 打分, ’维度二’: 打分, ..., ’综合得分’: 打分}，例如：{’事实正确性’: 9, ’满足用户需求’: 6, ’综合得分’: 7}。
    用户的提问：''',
    "Text_Assistant":'''请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是[处理文本事务]，因此你需要从下面的5个维度对回答进行评估: 
    1 清晰度(Clarity)
    是否表达清晰易懂，是否使用了简洁的语言和结构，以便用户可以轻松理解
    2 满足用户需求(User Satisfaction)
    是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应
    3 逻辑连贯性(Logical Coherence)
    是否在整体上保持一致，是否在不同部分之间保持逻辑连贯性，避免了自相矛盾
    4 事实正确性(Factuality)
    提供的信息是否准确无误，是否基于可信的事实和数据
    5 创造性(Creativity)
    是否具有创新性或独特性，是否提供了新颖，有文采的内容
    注意，回答不是越长越好，简短并且满足上述要求的回答是最好的。

    我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程：
    1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。
    2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个1～10的分数。
    3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
    4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。
    其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。

    当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；
    当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；
    当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；
    当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；
    只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。 
    作为示例，参考答案可以得到8分。

    请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数： 
    {’维度一’: 打分, ’维度二’: 打分, ..., ’综合得分’: 打分}，例如：{’清晰度’: 9, ’满足用户需求’: 6, ...’综合得分’: 7}。
    用户的提问：'''
    }

    grading_criteria = prompt_dict[intent]
    final_prompt = cfg.task.prompt.base.replace("TEMPLATE_GRADING_CRITERIA", grading_criteria)

    final_prompt = final_prompt.replace("TEMPLATE_QUESTION", question)
    final_prompt = final_prompt.replace("TEMPLATE_TEST_ANSWER", test_answer)

    final_prompt = final_prompt.replace("TEMPLATE_GPT4_ANSWER", gpt4_answer)

    return final_prompt


def _extract_balanced_candidates(text: str):
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        starts = [i for i, ch in enumerate(text) if ch == opener]
        for start in starts:
            depth = 0
            in_str = False
            escape = False
            quote_char = ""
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == quote_char:
                        in_str = False
                    continue
                if ch in ("'", '"'):
                    in_str = True
                    quote_char = ch
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:idx + 1].strip())
                        break
    candidates.sort(key=len, reverse=True)
    return candidates


def _try_parse_json_like(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass
    return None


def _json_obj_to_scoring_text(obj):
    if isinstance(obj, dict):
        for key in (
            "summarised_final_answer",
            "summary",
            "answer",
            "response",
            "final_answer",
            "output",
            "content",
            "text",
        ):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # Avoid leaking entire JSON blobs into scoring prompts.
        return ""
    if isinstance(obj, list):
        parts = [x.strip() for x in obj if isinstance(x, str) and x.strip()]
        if parts:
            return "\n".join(parts)
        return ""
    return str(obj)


def normalize_response_for_scoring(response):
    """Normalize potentially noisy JSON-like model output into scoreable text."""
    if response is None:
        return None
    if isinstance(response, (dict, list)):
        return _json_obj_to_scoring_text(response)
    if not isinstance(response, str):
        return None

    text = response.strip()
    if not text:
        return ""

    # Keep only content after known thought/marker prefixes.
    lower_text = text.lower()
    for marker in ("assistantfinal", "assistantanalysis"):
        marker_idx = lower_text.rfind(marker)
        if marker_idx != -1:
            text = text[marker_idx + len(marker):].strip()
            lower_text = text.lower()

    # Attempt direct parse.
    parsed = _try_parse_json_like(text)
    if parsed is not None:
        return _json_obj_to_scoring_text(parsed)

    # Try fenced blocks.
    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    for match in fence_pattern.finditer(text):
        block = match.group(1).strip()
        parsed = _try_parse_json_like(block)
        if parsed is not None:
            return _json_obj_to_scoring_text(parsed)

    # Try balanced JSON-like substrings embedded in text.
    for candidate in _extract_balanced_candidates(text):
        parsed = _try_parse_json_like(candidate)
        if parsed is not None:
            return _json_obj_to_scoring_text(parsed)

    # If it still looks like a JSON blob but we couldn't extract summary text,
    # do not feed raw structured text into scoring prompts.
    stripped = text.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or "summarised_final_answer" in stripped:
        return ""

    return text

def test_time_urs_scoring(cfg: DictConfig):

    print(f"Testing URS Scoring for {cfg.task.name}")
    df = pd.read_csv(cfg.task.dataset.path)
    prompt_list = []
    final_result_list = []
    number_of_answers_per_query = None
    for index, row in df.iterrows():
        question = row["question"]
        gpt4_answer = row["reference_answer"]
        test_answer = row["Response"]
        if "number_of_answers_per_query" in row:
            number_of_answers_per_query = row["number_of_answers_per_query"]
        test_answer = normalize_response_for_scoring(test_answer)
        if test_answer is None or not isinstance(test_answer, str):
            continue
        intent = row["intent"]
        prompt = urs_scoring_prompt(cfg, question, gpt4_answer, test_answer, intent)
        prompt_list.append(prompt)
        if number_of_answers_per_query is not None:
            print("number_of_answers_per_query:", number_of_answers_per_query)
            final_result_list.append({
                "question": question,
                "prompt": prompt,
                "reference_answer": gpt4_answer,
                "response": test_answer,
                "intent": intent,
                "number_of_answers_per_query": number_of_answers_per_query
            })
        else:
            final_result_list.append({
                "question": question,
                "prompt": prompt,
                "reference_answer": gpt4_answer,
                "response": test_answer,
                "intent": intent,
            })
    print(f"Total number of prompts: {len(prompt_list)}")
    print(f"First 5 prompts:")
    for prompt in prompt_list[:5]:
        print(prompt)
        print("-"*100)

    initialise_method = hydra.utils.get_method(cfg.inference.initialise_method)
    model, tokenizer = initialise_method(cfg)
    inference_method = hydra.utils.get_method(cfg.inference.inference_method)
    results = inference_method(model, tokenizer, cfg, prompt_list, final_result_list)
    for i, result in enumerate(results):
        final_result_list[i]["Response"] = result
        try:
            response_parser = hydra.utils.get_method(cfg.inference.response_parser)
            final_result_list[i]["Response"] = response_parser(result)
        except:
            pass
    if os.path.exists(cfg.experiment.output_dir) is False:
        os.makedirs(cfg.experiment.output_dir)
    print("Inference Results Dumped to", cfg.experiment.output_dir + "/inference_results.csv")
    pkl.dump(final_result_list, open(cfg.experiment.output_dir + "/inference_results.pkl", "wb"))
    pd.DataFrame(final_result_list).to_csv(cfg.experiment.output_dir + "/inference_results.csv", index=False)