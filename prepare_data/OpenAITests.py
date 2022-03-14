import glob
import os.path
import json
import nltk
import urllib
import urllib.request
import requests
import subprocess
from nltk.translate.bleu_score import sentence_bleu
import logging
import openai

class OpenAITests():
    """ Run tests on OpenAI:
        - since it seems that OpenAI at the moment lacks the ability to make batch completions (via an API/SDK or otherwise)
        -  we have to run predictions on test data one at a time and write them out to a separate file for 
    """
    def __init__(self, test_file_path):
        completion_suffix = " END"
        prompt_prefix = "\n\n###\n\n"
        prompt_suffix = "\n\n===\n\n"
        
        test_file = open(test_file_path, 'r')
        test_file_lines = test_file.readlines()

        for test_file_line in test_file_lines:
            test_file_line_json = json.loads(test_file_line)

            # print("Prompt: ", test_file_line_json['prompt'])
            # open_ai_output = input("Output of OpenAI API: ")

            # clean up prompt data
            prompt = test_file_line_json['prompt'].replace(prompt_prefix, "").replace(prompt_suffix, "")

            # import http.client as http_client
            # http_client.HTTPConnection.debuglevel = 1

            # # You must initialize logging, otherwise you'll not see debug output.
            # logging.basicConfig()
            # logging.getLogger().setLevel(logging.DEBUG)
            # requests_log = logging.getLogger("requests.packages.urllib3")
            # requests_log.setLevel(logging.DEBUG)
            # requests_log.propagate = True
            
            # url = "https://api.openai.com/v1/engines/davinci:ft-personal-2022-02-26-15-30-06/completions"
            
            # headers = {
            #             'Authorization': 'Bearer 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',
            #             'Content-type': 'application/json'
            #           }
            
            # payload = {
            #             'prompt': prompt,
            #             'model': 'davinci:ft-personal-2022-02-26-15-30-06', 
            #             'temperature': 0, 
            #             'max_tokens': 1025
            #           }

            # print(payload)
            # response = requests.post(url, headers = headers, data = payload)
            # print(response.text)
            obj_open_ai_output = openai.Completion.create(
                model = 'ada:ft-personal-2022-03-13-02-09-40',
                temperature = 0,
                max_tokens = 1025,
                prompt = prompt)

            open_ai_output = str(obj_open_ai_output.choices[0].text)
            print("Output "  + open_ai_output)
            # trim output at "END" delimiter - filler tokens should be trimmed out
            if open_ai_output.find(completion_suffix) > 0:
                open_ai_output = open_ai_output[:open_ai_output.find(completion_suffix)]
    
            print("Trimmed Output: " + open_ai_output)

            open_ai_output = open_ai_output + completion_suffix
            data_line = { "prompt" : test_file_line_json['prompt'], "truth": test_file_line_json['truth'], "openai_output": open_ai_output}

            test_ready_to_score_file = open("test_ready_to_score_file.json", "a")
            data_line_string = json.dumps(data_line) + "\n"
            write_it = test_ready_to_score_file.write(data_line_string)
            test_ready_to_score_file.close()
        
        test_file.close()
    
    def bleu_scores(test_scored_file_path):
        scores_file = open(test_scored_file_path, 'r')
        scores_lines = scores_file.readlines()

        lowest_score = 1
        highest_score = 0
        sum_of_scores = 0
        total_samples = 0

        for score_line in scores_lines:
            scored_line_json = json.loads(score_line)
            reference = scored_line_json["truth"].split()
            candidate = scored_line_json["openai_output"].split()
            bleu_score = sentence_bleu([reference], candidate)

            if bleu_score > highest_score:
                highest_score = bleu_score
            
            if bleu_score < lowest_score:
                lowest_score = bleu_score

            sum_of_scores += bleu_score
            total_samples += 1
            print("BLEU score for: ", scored_line_json['prompt'], " is ", bleu_score)

        print("Highest BLEU score: ", highest_score * 100, "%")
        print("Lowest BLEU score: ", lowest_score * 100, "%")
        print("Aggregate BLEU score: ", sum_of_scores/total_samples * 100, "%")
