from sklearn.feature_extraction.text import CountVectorizer
import os

negative_real_path = 'op_spam_v1.4/negative_polarity/deceptive_from_MTurk'

# negative_false_path = 'op_spam_v1.4/negative_polarity/deceptive_from_MTurk'

folder_path = ""

files_array = []
for folder in os.listdir(negative_real_path):
    print(folder)

# for filename in os.listdir(negative_real_path):
#     if filename.endswith('.txt'):
#         with open(os.path.join(negative_real_path, filename), encoding='utf-8') as f:
#             files_array.append(f.read())

# print(files_array)