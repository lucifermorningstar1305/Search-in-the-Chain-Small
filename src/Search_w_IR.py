import numpy as np
import json
import bitsandbytes
import torch
import torch.nn as nn
import socket
import os
import sys
import logging
import traceback

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig
)

from dotenv import load_dotenv

def execute(data_path: str, start_idx: int, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> int:
    """
    This function is used to perform the Search in the Chain for 
    the Language Model.

    :param data_path: the data path
    :param model: a hugging face AutoModelForCausalLM object
    :param tokenizer: a hugging face AutoTokenizer object

    :returns: an integer value to stop the execution
    """

    data = open(data_path, "r")

    generation_config = model.generation_config
    generation_config.max_new_tokens = 64
    generation_config.temperature = 1.
    generation_config.top_p = 1.
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id

    for k, example in enumerate(data):
        if k < start_idx:
            continue
        

        example = json.loads(example)
        ques = example["question"]
        answer = example["answer"]

        logging.info(f"loading example: {ques}--{answer}")
        round_count = 0

        prompt = f"""
        Construct a global reasoning chain for this complex [Question] : " {ques} " You should generate a query to the search engine based on what you already know at each step of the reasoning chain, starting with [Query]. If you know the answer for [Query], generate it starting with [Answer].
        You can try to generate the final answer for the [Question] by referring to the [Query]-[Answer] pairs, starting with [Final Content].
        If you don't know the answer, generate a query to search engine based on what you already know and do not know, starting with [Unsolved Query].
        For example:
        [Question]: "Where do greyhound buses that are in the birthplace of Spirit If...'s performer leave from? "
        [Query 1]: Who is the performer of Spirit If... ?
        If you don't know the answer:
        [Unsolved Query]: Who is the performer of Spirit If... ?
        If you know the answer:
        [Answer 1]: The performer of Spirit If... is Kevin Drew.
        [Query 2]: Where was Kevin Drew born?
        If you don't know the answer:
        [Unsolved Query]: Where was Kevin Drew born?
        If you know the answer:
        [Answer 2]: Toronto.
        [Query 3]: Where do greyhound buses in Toronto leave from?
        If you don't know the answer:
        [Unsolved Query]: Where do greyhound buses in Toronto leave from?
        If you know the answer:
        [Answer 3]: Toronto Coach Terminal.
        [Final Content]: The performer of Spirit If... is Kevin Drew [1]. Kevin Drew was born in Toronto [2]. Greyhound buses in Toronto leave from Toronto Coach Terminal [3]. So the final answer is Toronto Coach Terminal.
                                
        [Question]:"Which magazine was started first Arthur’s Magazine or First for Women?"
        [Query 1]: When was Arthur’s Magazine started?
        [Answer 1]: 1844.
        [Query 2]: When was First for Women started?
        [Answer 2]: 1989
        [Final Content]: Arthur’s Magazine started in 1844 [1]. First for Women started in 1989 [2]. So Arthur’s Magazine was started first. So the answer is Arthur’s Magazine

        [Question]: {ques}"""
        
        feedback_answer = "continue"
        predict_answer = ""
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 50007))

        device = "cuda:1"

        while round_count < 5 and not feedback_answer == "end":    
            try:
                encoding = tokenizer(prompt, return_tensors = "pt").to(device)
                
                logging.info("Quering Falcon")
                with torch.inference_mode():
                    outputs = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask, generation_config=generation_config)
                
                resp = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt)-2:]
                predict_answer += resp

                logging.info("Sending generated answer to Colbert")
                sock.send(resp[len(prompt)-2:].encode())

                feedback = sock.recv(10240).decode()
                logging.info(f"Recieved feedback from Colbert: {feedback}")

                if feedback == "end":
                    break
                
                feedback_list = feedback.split('<SEP>')

                if not 'Unsolved Query' in feedback:
                    new_prompt = """
                    According to this Reference, the answer for "{}" should be "{}",  
                    you can change your answer based on the Reference and continue constructing the reasoning chain to give the final answer for [Question]:{}
                    Reference: {}
                    """.format(feedback_list[0],feedback_list[1],ques,feedback_list[2])
                else:
                    new_prompt = """
                    According to this Reference, the answer for "{}" should be "{}",  
                    you can give your answer based on the Reference and continue constructing the reasoning chain to give the final answer for [Question]：{}
                    Reference: {}
                    """.format(feedback_list[0],feedback_list[1],ques,feedback_list[2])   

                prompt += f"\n{new_prompt}"
                round_count += 1

            except Exception as e:
                logging.error(f"Start idx: {k}")
                sock.send('end'.encode())
                sock.close()
                return k
            
        if not feedback_answer == 'end':
            sock.send('end'.encode())
        sock.close()        
    
    return -1

if __name__ == "__main__":

    load_dotenv()
    logging.basicConfig(filename="../logs/Search_w_IR.log",
                        level=logging.DEBUG,
                        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    MODEL_NAME = "tiiuae/falcon-7b-instruct"
    
    try:
        logging.info("Initializing model")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config
        )
        
        logging.info("Model initialization done")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        logging.info("Starting execution")

        start_idx = 0

        while not start_idx == -1:
            start_idx = execute("../data/hotpotqa/hotpot_dev_fullwiki_v1.json",
                                start_idx, model, tokenizer)

    except Exception as e:
        logging.error(traceback.format_exc())

