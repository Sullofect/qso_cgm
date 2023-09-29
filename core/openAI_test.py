import os
import numpy
import openai
import matplotlib.pyplot as plt


openai.organization = "org-JfFERCj4AqGYGLseeA1YgPy6"
openai.api_key = open("/Users/lzq/Dropbox/API_info.rtf", "r").read()
ls = openai.Model.list()
print(ls)
openai.Image.create(
  prompt="A cute baby sea otter",
  n=2,
  size="1024x1024"
)
# response = openai.Image.create(prompt="a white siamese cat", n=1, size="1024x1024")


# plt.figure()
# plt.imshow(response['data'])
# plt.show()
# image_url = response['data'][0]['url']