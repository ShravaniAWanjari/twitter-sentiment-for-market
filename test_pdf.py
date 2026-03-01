import requests
url = "http://localhost:8000/api/pdf"
try:
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Content Length: {len(response.content)}")
        with open("test_output.pdf", "wb") as f:
            f.write(response.content)
        print("PDF saved as test_output.pdf")
    else:
        print(f"Body: {response.text}")
except Exception as e:
    print(f"Error: {e}")
