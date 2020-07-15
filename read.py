from operator import itemgetter
from collections import defaultdict
import jsonlines
import os
import shutil
from tqdm import tqdm

def dict_to_list(dict):
    list = []
    for key, value in dict.items():
        temp_pair = [key, value]
        list.append(temp_pair)
    list = sorted(list, key=itemgetter(1))
    list = list[::-1]
    return list

def write_file(directory_stem, folder, obj):
    new_path = directory_stem + '\\' + '{}'.format(folder)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    file_to_write_name = str(obj['id'])
    with open('{}.txt'.format(file_to_write_name), 'w', encoding='utf-8') as f:
        print(str(obj['body']), file=f)
    source = directory_stem + '\{}.txt'.format(file_to_write_name)
    dest = directory_stem + '\{}\{}.txt'.format(folder, file_to_write_name)
    os.rename(source, dest)


filename = 'aylien_covid_news_data.jsonl'
directory_stem = r'C:\Users\ascam\PycharmProjects\polarizat'
source_count = defaultdict(int)
with jsonlines.open(filename, 'r') as fp:
    counter = 0
    for obj in tqdm(fp):
        source_count[obj['source']['domain']] += 1
        '''
        if obj['source']['domain'] == 'reuters.com':
            write_file(directory_stem, 'reuters', obj)
        elif obj['source']['domain'] == 'foxnews.com':
            write_file(directory_stem, 'foxnews', obj)
        elif obj['source']['domain'] == 'breitbart.com':
            write_file(directory_stem, 'breitbart', obj)
        elif obj['source']['domain'] == 'cnn.com':
            write_file(directory_stem, 'cnn', obj)
        elif obj['source']['domain'] == 'huffingtonpost.com':
            write_file(directory_stem, 'huffpo', obj)
        '''
        counter += 1
source_count_list = dict_to_list(source_count)
with open('source_count_list', 'w', encoding='utf-8') as fp:
    print(*source_count_list, sep='\n', file=fp)
print('done')
