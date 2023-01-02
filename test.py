import openai
import requests

openai.api_key = "sk-rXwHNrNvXmaQv9n9OKe3NPCBvH7RGXhCmK3YQNRm"
response = openai.Image.create(
    prompt="a white siamese cat",
    n=1,
    size="512x512"
)
image_url = response['data'][0]['url']
img_data = requests.get(image_url).content
with open('image_name.jpg', 'wb') as handler:
    handler.write(img_data)