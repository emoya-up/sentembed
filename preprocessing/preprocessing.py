import os
import io
import re
import csv
import json
import tarfile
import numpy as np

def write_any_parallel(path: str, field_names: list, lines: dict):
    '''
    yields a file of 1-1 aligned text segments
    args:
        path: path to save to
        field_names: names of languages and/or text witnesses for dictionary
        lines: line-based dictionary
        
    return:
        None
    '''
    with open(path, mode='w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, field_names)
        output_dict = {name : [] for name in field_names}
        for i in range(len(lines['en'])):
            for name in field_names:
                output_dict[name] = lines[name][i]
            writer.writerow(output_dict)

file_name = "C:/Users/Administrator/OneDrive/Cognitive Systems/WS2425/3_Execution/PM1_Opinion_argument_mining/parallel_corpus/6way/data/6way.tar"
file_path = 'data/'

if os.path.isfile('data/mask.json'): # mask based on one text witness
    with open('data/mask.json') as json_mask:
        contains_speech = json.load(json_mask)
else:
    contains_speech = []
    
line_dict = {}
with tarfile.TarFile(file_name, mode='r', encoding='utf-8') as archive:
    for tarinfo in archive.getmembers():
        # reset skip for fatal errors
        skip = False
        
        # if file is among relevant language files
        if tarinfo.name[-3:] not in ['ids', 'DME', 'MER', 'way', 'pdf', '.ar', '.zh']:
            print(tarinfo.name)
            ln = tarinfo.name[-2:]
            
            sent_n = 0
            
            # write content to file
            content = archive.extractfile(tarinfo)
                    
            # default case for all languages including english
            # if contains_speech has not yet been filled, append truth values for all lines that outwardly appear to be parts of speeches
            if ln == 'en' and len(contains_speech) == 0:
                for line in content:
                    line = line.decode('utf-8')
                    # sentences = list(tokenizer.sentence_tokenize(line, use_abbreviation=True))
                    # for sent in sentences:
                    if not re.match(r'\d+\.(\d*\.*)*|RESOLUTION|CONTENTS|(a-zA-Z)\)', line[:10]):
                        contains_speech.append(True)
                    else:
                        contains_speech.append(False)
                print(len(contains_speech))
                with open('data/mask.json', 'w') as json_mask:
                    json.dump(contains_speech, json_mask)
                    
            # default case for all languages including english
            else:
                line_dict[ln] = []
                with open(file_path + 'UNPD_' + ln + '.txt', 'w', encoding='utf-8') as file:
                    for line in content:
                        line = line.decode('utf-8')
                        try:
                            if contains_speech[sent_n]:
                                file.write(line)
                                line_dict[ln].append(line)
                            sent_n += 1
                        except IndexError:
                            print('IndexError at line ' + str(sent_n))
                            skip = True
                            break
                        except Exception as e:
                            print(e)
                            skip = True
                            break
                    
                # abort language processing in case of fatal error
                if skip:
                    continue          
                  
        # print number of sentences when finished
            print(sent_n)

write_any_parallel('data/sent_text_all.csv', ['en', 'es', 'fr', 'ru'], line_dict)