# Dictionary encoding and decoding for int

def dictionaryEncoding(input_list):
    num_dict = {}
    nums = []
    indexs = []
    for num in input_list:
        if num not in num_dict:
            nums.append(num)
            num_dict[num] = len(nums) - 1
        indexs.append(num_dict[num])
    return nums, indexs


def dictionaryDecoding(nums, indexs):
    return [nums[i] for i in indexs]


if __name__ == "__main__":
    input_list = [3, 4, 3, 3, 3, 5, 4]
    nums, indexs = dictionaryEncoding(input_list)
    print(nums, indexs)
    decoded_result = dictionaryDecoding(nums, indexs)
    print(decoded_result)

# class DictionaryInt:
#     def __init__(self):
#         self.dictionary = {}
#         self.dictionaryList = []

#     # 根据列表的字典信息恢复字典
#     def loadDictionary(self, dictionary_list):
#         for i in range(1, len(dictionary_list)):
#             self.dictionaryList.append(dictionary_list[i])
#             self.dictionary[dictionary_list[i]] = i-1

#     # 编码一个value，得到对应的字典中的index
#     def dictionaryEncoding(self, value):
#         if value not in self.dictionary:
#             self.dictionary[value] = len(self.dictionary)
#             self.dictionaryList.append(value)
#         return self.dictionary[value]

#     # 解码一个index，得到原本的value
#     def dictionaryDecoding(self, index):
#         return self.dictionaryList[index]

#     def dictionaryEncodings(self, values):
#         indexs = []
#         for value in values:
#             indexs.append(self.dictionaryEncoding(value))
#         return indexs

#     def dictionaryDecodings(self, indexs):
#         values = []
#         for index in indexs:
#             values.append(self.dictionaryList[index])
#         return values

#     # 将字典信息编码
#     def flushDictionary(self):
#         dic = []
#         dic.append(len(self.dictionaryList))
#         for value in self.dictionaryList:
#             dic.append(value)
#         return dic
