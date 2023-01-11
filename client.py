import requests

data = {
    # "audio": "https://replicate.delivery/pbxt/I62Y7Lf1KrjU2XRxXdBSQx0UulZQnaayiYbr6NfzaEaZrhh5/2023-01-06%2019.25.47.ogg",
    "audio": "https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-0.mp3",
    "language": "en"
}
response = requests.post("http://localhost:8000", json=data)
print(response.json())