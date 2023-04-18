import torch
from llama_index import WikipediaReader

import pandas as pd



def csv_prompter(generator, tokenizer, question):


    fulltext = "A question is provided below. Given the question, extract " + \
               "keywords from the text. Focus on extracting the keywords that we can use " + \
               "to best lookup answers to the question. \n" + \
               "---------------------\n" + \
               "{}\n".format(question) + \
               "---------------------\n" + \
               "Provide keywords in the following comma-separated format.\nKeywords: "

    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        generated_ids = generator(
            gen_in,
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.1,  # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
            temperature=0.5,  # default: 1.0
            top_k=50,  # default: 50
            top_p=1.0,  # default: 1.0
            early_stopping=True,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0]  # for some reason, batch_decode returns an array of one element?
        text_without_prompt = generated_text[len(fulltext):]
    response = text_without_prompt
    response = response.split("===")[0]
    response.strip()
    print(response)
    keywords = response.split(", ")
    print(keywords)

    df = pd.read_csv('disease_symptom.csv')
    divided_text = []
    csvdata = df.to_dict('records')
    for csv_item in range(0,len(csvdata),6):
        csv_text = str(csvdata[csv_item:csv_item+6]).replace("}, {", "\n\n").replace("[", "").replace("]", "").replace("\"", "")
        divided_text.append(csv_text)

    answer_llama = ""

    score_textlist = [0] * len(divided_text)

    for i, chunk in enumerate(divided_text):
        for t, keyw in enumerate(keywords):
            if keyw.lower() in chunk.lower():
                score_textlist[i] = score_textlist[i] + 1

    answer_list = []
    divided_text = [item for _, item in sorted(zip(score_textlist, divided_text), reverse=True)]
    divided_text.append("_")
    for i, chunk in enumerate(divided_text):

        if i < 4 and not i == int(len(divided_text) - 1):
            fulltext = "{}".format(chunk) + \
                       "\n---------------------\n" + \
                       "Based on the diseases and corresponding symptoms in the Table above, " + \
                       "answer the question: {}\n".format(question) + \
                       "Disease name and corresponding symptoms: "
        elif i == int(len(divided_text) - 1) and len(answer_list) > 1:
            fulltext = "The original question is as follows: {}\n".format(question) + \
                       "We have provided existing answers:\n" + \
                       "------------\n" + \
                       "{}\n".format(str("\n\n".join(answer_list))) + \
                       "------------\n" + \
                       "The best one answer: "
        else:
            continue

        print(fulltext)
        gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()

        with torch.no_grad():
            generated_ids = generator(
                gen_in,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1,  # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=0.5,  # default: 1.0
                top_k=50,  # default: 50
                top_p=1.0,  # default: 1.0
                early_stopping=True,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text_without_prompt = generated_text[len(fulltext):]

        answer_llama = text_without_prompt
        print()
        print("\nAnswer: " + answer_llama)
        print()
        answer_list.append(answer_llama)

    return answer_llama


